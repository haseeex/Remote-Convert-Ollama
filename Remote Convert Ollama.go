/*
接口提供的字段兼容性优先适配 Visual Studio Code Copilot Chat
https://github.com/microsoft/vscode-copilot-chat/blob/main/src/extension/byok/vscode-node/ollamaProvider.ts#L137C2-L137C124
Ollama 官方文档关于思考模式的说明
https://docs.ollama.com/capabilities/thinking
*/
package main

import (
	"bufio"
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	_ "embed"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

type ModelDetailedSetting struct {
	ContextLength   int64    `json:"ContextLength"`
	MaxOutputTokens int64    `json:"MaxOutputTokens"`
	Capabilities    []string `json:"Capabilities,omitempty"`
	// VisionProxyModel 视觉代理模型：主模型不支持图片时，指定一个支持视觉的上游模型 ID，
	// 收到图片请求时先用它识别图片，再把识别文本合并进主模型请求（图片本身不再转发给主模型）
	VisionProxyModel string `json:"VisionProxyModel,omitempty"`
	// VisionProxyPrompt 视觉代理自定义提示词：帮助代理模型更好地识别图片（可留空使用默认提示词）
	VisionProxyPrompt string `json:"VisionProxyPrompt,omitempty"`
}

// PromptReplaceRule 定义请求提示词替换规则
type PromptReplaceRule struct {
	Enable  bool   `json:"enable"`
	Mode    string `json:"mode,omitempty"`  // 替换模式枚举：normal=普通替换（默认）/ whole=匹配整段替换 / force=强制替换
	Index   *int   `json:"index,omitempty"` // 消息索引（nil=未指定, 0=第1条），与 role 配合使用
	Role    string `json:"role,omitempty"`  // 按角色定位（如 "system"）
	Prompt  string `json:"prompt"`
	Replace string `json:"replace"`
}

// 替换模式枚举常量
const (
	replaceModeNormal = "normal" // 普通替换：仅替换 prompt 匹配片段
	replaceModeWhole  = "whole"  // 匹配整段替换：匹配到 prompt 后将整条消息替换为 replace
	replaceModeForce  = "force"  // 强制替换：不检查匹配，直接整条替换
)

// normalizeReplaceMode 规范化替换模式枚举，非法值回退为 normal
func normalizeReplaceMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case replaceModeForce:
		return replaceModeForce
	case replaceModeWhole:
		return replaceModeWhole
	default:
		return replaceModeNormal
	}
}

// migrateRequestPromptReplace 将旧版替换规则的 force / replaceWhole 布尔字段迁移为 mode 枚举。
// 返回是否发生了迁移（需要写回 config.json）。
// 旧字段已被 json.Unmarshal 丢弃，因此从原始 rawMap 读取。
// 兼容规则：force=true → mode=force；否则 replaceWhole=true → mode=whole；其余保持 normal。
func migrateRequestPromptReplace(stored *Config, rawMap map[string]interface{}) bool {
	rawRules, _ := rawMap["RequestPromptReplace"].(map[string]interface{})
	migrated := false
	for name, rule := range stored.RequestPromptReplace {
		// 规则已显式指定 mode → 仅规范化
		if strings.TrimSpace(rule.Mode) != "" {
			rule.Mode = normalizeReplaceMode(rule.Mode)
			stored.RequestPromptReplace[name] = rule
			continue
		}
		// 从原始 JSON 读取旧布尔字段
		mode := replaceModeNormal
		if rawRule, ok := rawRules[name].(map[string]interface{}); ok {
			if f, _ := rawRule["force"].(bool); f {
				mode = replaceModeForce
			} else if w, _ := rawRule["replaceWhole"].(bool); w {
				mode = replaceModeWhole
			}
		}
		if mode != replaceModeNormal {
			rule.Mode = mode
			stored.RequestPromptReplace[name] = rule
			migrated = true
		}
	}
	return migrated
}

type Config struct {
	IP                    string                          `json:"IP"`
	PORT                  string                          `json:"PORT"`
	Log_Limit             int64                           `json:"Log_Limit"`
	Log_Responses         bool                            `json:"Log_Responses"`
	Log_Headers           bool                            `json:"Log_Headers"`
	Log_Body              bool                            `json:"Log_Body"`
	OpenAIPrefix          string                          `json:"OpenAI_Prefix"`
	OpenAISuffix          string                          `json:"OpenAI_Suffix"`
	StreamMode            string                          `json:"StreamMode"`
	Capabilities          []string                        `json:"Capabilities"`
	OpenAIBase            string                          `json:"OPENAI_BASE"`
	OpenAIKey             string                          `json:"OPENAI_KEY"`
	WebConfigPassword     string                          `json:"WebConfigPassword,omitempty"` // 配置管理页面访问密码（空=不启用）
	ModelAlias            map[string]string               `json:"ModelAlias"`
	ModelDetailedSettings map[string]ModelDetailedSetting `json:"ModelDetailedSettings"`
	RequestPromptReplace  map[string]PromptReplaceRule    `json:"RequestPromptReplace,omitempty"`
	// VisionProxyPrompt 全局默认视觉代理提示词：模型未自定义 VisionProxyPrompt 时使用（留空=内置默认提示词）
	VisionProxyPrompt string `json:"VisionProxyPrompt,omitempty"`
}

var requestCount int64
var clear map[string]func() //创建一个用于存储清除函数的映射

var cfg Config

// storedOpenAIKey 保存 config.json 中持久化的 OPENAI_KEY（加密格式），
// 供配置管理页面在用户未修改密钥时原样写回，避免密钥丢失
var storedOpenAIKey string

// storedWebConfigPassword 保存 config.json 中持久化的 WebConfigPassword（加密格式），
// 供配置管理页面在用户未修改密码时原样写回，避免密码丢失
var storedWebConfigPassword string

// lastReasoningContent 存储上一个 assistant 响应的 reasoning_content，
// 因为 DeepSeek 思考模式要求客户端下次请求时必须传回这个字段。
// 如果 Cherry Studio 没有自动带上，我们就在转发时自动注入。
var (
	lastReasoningContent string
	rcMutex              sync.RWMutex
)

// setLastReasoningContent 线程安全地设置 lastReasoningContent
func setLastReasoningContent(rc string) {
	rcMutex.Lock()
	lastReasoningContent = rc
	rcMutex.Unlock()
}

// getLastReasoningContent 线程安全地获取 lastReasoningContent
func getLastReasoningContent() string {
	rcMutex.RLock()
	defer rcMutex.RUnlock()
	return lastReasoningContent
}

const encryptedKeyPrefix = "已加密|"

const (
	streamModePreserve    = "preserve"
	streamModeForceStream = "force_stream"
	streamModeForceClose  = "force_close"
)

// 这个 UUID 是用来增强加密安全性的，确保同一台机器上的加密结果不同于其他机器。它不会泄露任何敏感信息。
// 推荐生成网站 https://www.uuidgenerator.net/ 生成一个随机的 UUID 来替换这个值。
const secretUUID = "vancat-10a8bca6-fe6f-4bcd-8c9a-9a27d6ec1b16"

// 上下文默认值，当上游 API 未返回模型元数据时使用
const DefaultContextLength = 1000000
const DefaultMaxOutputTokens = 384000

// OllamaToolCall 是 Ollama 格式的工具调用
// ID 非标准 Ollama 字段（omitempty），用于在响应中保留上游 tool_call id：
// 客户端回传 assistant 消息时可直接配对，避免 tool 消息与 tool_calls 关联断裂导致 Agent 死循环
type OllamaToolCall struct {
	ID       string `json:"id,omitempty"`
	Function struct {
		Name      string      `json:"name"`
		Arguments interface{} `json:"arguments"` // JSON 对象（非字符串）
	} `json:"function"`
}

// OpenAIToolCall 是 OpenAI 格式的工具调用
type OpenAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // JSON 字符串
	} `json:"function"`
}

// Choice 用于非流式响应的 choice
type Choice struct {
	Index        int                    `json:"index"`
	Message      map[string]interface{} `json:"message"`
	Text         string                 `json:"text"`
	Delta        map[string]interface{} `json:"delta,omitempty"`
	FinishReason *string                `json:"finish_reason"`
}

// OpenAIResp 用于非流式响应
type OpenAIResp struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

// -------------------- Anthropic Messages API 类型 --------------------

type AnthropicReq struct {
	Model         string                 `json:"model"`
	Messages      []AnthropicMessage     `json:"messages"`
	System        interface{}            `json:"system,omitempty"` // string 或 []{type, text}
	MaxTokens     int                    `json:"max_tokens"`
	Stream        bool                   `json:"stream,omitempty"`
	Temperature   *float64               `json:"temperature,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	Tools         []interface{}          `json:"tools,omitempty"`
	ToolChoice    interface{}            `json:"tool_choice,omitempty"`
}

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string 或 []AnthropicContentBlock
}

type AnthropicContentBlock struct {
	Type   string              `json:"type"`
	Text   string              `json:"text,omitempty"`
	Source *AnthropicImgSource `json:"source,omitempty"` // 图片块
}

type AnthropicImgSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type AnthropicResp struct {
	ID           string               `json:"id"`
	Type         string               `json:"type"`
	Role         string               `json:"role"`
	Content      []AnthropicRespBlock `json:"content"`
	Model        string               `json:"model"`
	StopReason   string               `json:"stop_reason"`
	StopSequence *string              `json:"stop_sequence"`
	Usage        AnthropicUsage       `json:"usage"`
}

type AnthropicRespBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// OpenAI 标准响应结构（含 usage）
type OpenAIUsageResp struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// OpenAI 流式 chunk
type OpenAIChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

// OpenAIDeltaToolCall 是流式 delta 中的 tool_call 片段
type OpenAIDeltaToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function"`
}

func extractContent(resp *OpenAIResp) string {
	if len(resp.Choices) == 0 {
		return ""
	}
	ch := resp.Choices[0]

	if msg, ok := ch.Message["content"].(string); ok && msg != "" {
		return msg
	}
	if ch.Text != "" {
		return ch.Text
	}
	if delta, ok := ch.Delta["content"].(string); ok && delta != "" {
		return delta
	}
	return ""
}

// extractToolCalls 从 OpenAI 响应中提取 tool_calls 并转为 Ollama 格式
// 保留上游 tool_call id（空则生成 call_N 兜底），客户端回传 assistant 消息时
// 可与 tool 消息的 tool_call_id 严格配对，避免 Agent 死循环
func extractToolCalls(msg map[string]interface{}) []OllamaToolCall {
	tcRaw, ok := msg["tool_calls"]
	if !ok {
		return nil
	}

	tcList, ok := tcRaw.([]interface{})
	if !ok {
		return nil
	}

	var result []OllamaToolCall
	seq := 0
	for _, tc := range tcList {
		tcMap, ok := tc.(map[string]interface{})
		if !ok {
			continue
		}
		// 提取 function
		funcRaw, ok := tcMap["function"]
		if !ok {
			continue
		}
		funcMap, ok := funcRaw.(map[string]interface{})
		if !ok {
			continue
		}

		var otc OllamaToolCall
		otc.ID, _ = tcMap["id"].(string)
		if otc.ID == "" {
			seq++
			otc.ID = fmt.Sprintf("call_%d", seq)
		}
		name, _ := funcMap["name"].(string)
		otc.Function.Name = name

		// arguments 在 OpenAI 中是 JSON 字符串，需要反序列化为对象
		if argsStr, ok := funcMap["arguments"].(string); ok {
			var argsObj interface{}
			if err := json.Unmarshal([]byte(argsStr), &argsObj); err == nil {
				otc.Function.Arguments = argsObj
			} else {
				otc.Function.Arguments = argsStr
			}
		} else {
			otc.Function.Arguments = funcMap["arguments"]
		}
		result = append(result, otc)
	}
	return result
}

// convertMessagesToOpenAI 将 Ollama 格式的消息列表转为 OpenAI 格式
// 关键：维护 tool_call_id 闭环 —— assistant.tool_calls[].id 与后续 tool 消息的
// tool_call_id 必须严格配对，否则上游 API 无法关联工具结果，
// 模型会认为"调用没有返回结果"而反复重试同一调用（Agent 死循环的根源）
func convertMessagesToOpenAI(messages []interface{}) []interface{} {
	var result []interface{}
	toolCallSeq := 0            // 本请求内 tool_call id 生成序号
	var pendingToolIDs []string // 待消费的 tool_call id 队列（assistant 声明 → tool 消息按序消费）

	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			result = append(result, msg)
			continue
		}

		role, _ := msgMap["role"].(string)

		// 处理 tool 角色消息：补齐 tool_call_id（按顺序消费前面 assistant 声明的 id）
		if role == "tool" {
			newMsg := make(map[string]interface{})
			for k, v := range msgMap {
				newMsg[k] = v
			}
			if id, hasID := newMsg["tool_call_id"].(string); !hasID || id == "" {
				if len(pendingToolIDs) > 0 {
					// 从队列头部取对应的 tool_call id
					newMsg["tool_call_id"] = pendingToolIDs[0]
					pendingToolIDs = pendingToolIDs[1:]
				} else {
					toolCallSeq++
					newMsg["tool_call_id"] = fmt.Sprintf("call_%d", toolCallSeq)
				}
			}
			result = append(result, newMsg)
			continue
		}

		// 非 tool 消息出现时清空待消费队列（assistant 无 tool_calls / user / system）
		if role != "assistant" {
			pendingToolIDs = nil
		}

		// 处理 assistant 消息中的 tool_calls（转换为 OpenAI 格式）
		if role == "assistant" {
			// DeepSeek 思考模式：如果客户端没带 reasoning_content，但我们有存储，就自动注入
			if _, hasRC := msgMap["reasoning_content"]; !hasRC && getLastReasoningContent() != "" {
				// 创建一个新 map 避免修改原始 msg 的 map
				newMsg := make(map[string]interface{})
				for k, v := range msgMap {
					newMsg[k] = v
				}
				newMsg["reasoning_content"] = getLastReasoningContent()
				msgMap = newMsg
			}

			tcRaw, hasTC := msgMap["tool_calls"]
			if !hasTC {
				// assistant 无 tool_calls：清空待消费队列，避免旧 id 错误配对到后面的 tool 消息
				pendingToolIDs = nil
				result = append(result, msgMap) // ← 修复：用 msgMap（可能含注入的 reasoning_content）
				continue
			}

			newMsg := make(map[string]interface{})
			for k, v := range msgMap {
				if k != "tool_calls" {
					newMsg[k] = v
				}
			}

			tcList, ok := tcRaw.([]interface{})
			if !ok {
				pendingToolIDs = nil
				result = append(result, msgMap) // ← 修复：用 msgMap
				continue
			}

			var openaiTCs []map[string]interface{}
			for _, tc := range tcList {
				tcMap, ok := tc.(map[string]interface{})
				if !ok {
					continue
				}

				// 检查是否已经有 id 字段 → 已经是 OpenAI 格式（仅需转换 arguments）
				if existingID, hasID := tcMap["id"].(string); hasID && existingID != "" {
					openaiTC := map[string]interface{}{
						"id":   existingID,
						"type": "function",
					}
					if funcRaw, ok := tcMap["function"]; ok {
						if funcMap, ok := funcRaw.(map[string]interface{}); ok {
							fn := map[string]interface{}{}
							if n, ok := funcMap["name"]; ok {
								fn["name"] = n
							}
							// arguments 是对象就序列化为字符串
							if args, ok := funcMap["arguments"]; ok {
								switch a := args.(type) {
								case string:
									fn["arguments"] = a
								default:
									argsBytes, _ := json.Marshal(a)
									fn["arguments"] = string(argsBytes)
								}
							}
							openaiTC["function"] = fn
						}
					}
					// 记录已声明的 id，供后续 tool 消息按序配对
					pendingToolIDs = append(pendingToolIDs, existingID)
					openaiTCs = append(openaiTCs, openaiTC)
				} else {
					// Ollama 格式（无 id）：生成确定性 id 并登记到待消费队列
					funcRaw, ok := tcMap["function"]
					if !ok {
						continue
					}
					funcMap, ok := funcRaw.(map[string]interface{})
					if !ok {
						continue
					}

					toolCallSeq++
					genID := fmt.Sprintf("call_%d", toolCallSeq)
					pendingToolIDs = append(pendingToolIDs, genID)

					openaiTC := map[string]interface{}{
						"id":   genID,
						"type": "function",
						"function": map[string]interface{}{
							"name": funcMap["name"],
						},
					}

					if args, ok := funcMap["arguments"]; ok {
						switch a := args.(type) {
						case string:
							openaiTC["function"].(map[string]interface{})["arguments"] = a
						default:
							argsBytes, _ := json.Marshal(a)
							openaiTC["function"].(map[string]interface{})["arguments"] = string(argsBytes)
						}
					}
					openaiTCs = append(openaiTCs, openaiTC)
				}
			}
			newMsg["tool_calls"] = openaiTCs
			// assistant 带 tool_calls 时不清空 pendingToolIDs（刚登记的 id 等待 tool 消息消费）
			result = append(result, newMsg)
			continue
		}

		// 处理 user/其他角色消息中的图片（Ollama 格式）
		// Ollama 支持两种图片格式：
		// 1. images 数组: {"role":"user","content":"文本","images":["base64..."]}
		// 2. content 数组中的 image 块: {"role":"user","content":[{"type":"image","image_base64":"base64..."}]}
		if role == "user" || role == "system" {
			newMsg := make(map[string]interface{})
			for k, v := range msgMap {
				newMsg[k] = v
			}

			// 情况1: 检查 images 数组字段
			hasImagesArray := false
			if imagesRaw, hasImages := newMsg["images"]; hasImages {
				if imagesList, ok := imagesRaw.([]interface{}); ok && len(imagesList) > 0 {
					hasImagesArray = true
					// 从 newMsg 移除 images 字段（OpenAI 不支持）
					delete(newMsg, "images")

					// 获取文本内容
					textContent, _ := newMsg["content"].(string)

					// 构建 content 数组
					var contentArray []interface{}
					if textContent != "" {
						contentArray = append(contentArray, map[string]interface{}{
							"type": "text",
							"text": textContent,
						})
					}
					for _, img := range imagesList {
						if imgStr, ok := img.(string); ok && imgStr != "" {
							mime := detectImageMIME(imgStr)
							contentArray = append(contentArray, map[string]interface{}{
								"type": "image_url",
								"image_url": map[string]interface{}{
									"url": "data:" + mime + ";base64," + imgStr,
								},
							})
						}
					}
					if len(contentArray) > 0 {
						newMsg["content"] = contentArray
					}
				}
			}

			// 情况2: 检查 content 是否为数组且包含 type:"image" 块
			if !hasImagesArray {
				if contentArray, ok := newMsg["content"].([]interface{}); ok {
					hasImageBlock := false
					for _, block := range contentArray {
						if blockMap, ok := block.(map[string]interface{}); ok {
							if t, _ := blockMap["type"].(string); t == "image" {
								hasImageBlock = true
								break
							}
						}
					}
					if hasImageBlock {
						var newContent []interface{}
						for _, block := range contentArray {
							if blockMap, ok := block.(map[string]interface{}); ok {
								if t, _ := blockMap["type"].(string); t == "image" {
									// Ollama image 块 → OpenAI image_url 块
									imgBase64, _ := blockMap["image_base64"].(string)
									if imgBase64 != "" {
										mime := detectImageMIME(imgBase64)
										newContent = append(newContent, map[string]interface{}{
											"type": "image_url",
											"image_url": map[string]interface{}{
												"url": "data:" + mime + ";base64," + imgBase64,
											},
										})
									}
								} else {
									newContent = append(newContent, block)
								}
							} else {
								newContent = append(newContent, block)
							}
						}
						if len(newContent) > 0 {
							newMsg["content"] = newContent
						}
					}
				}
			}

			result = append(result, newMsg)
			continue
		}

		result = append(result, msg)
	}
	return result
}

// hasToolCalls 判断响应中是否包含 tool_calls
func hasToolCalls(msg map[string]interface{}) bool {
	tc, ok := msg["tool_calls"]
	return ok && tc != nil
}

// detectImageMIME 根据 Base64 数据的前几个字符猜测图片 MIME 类型
func detectImageMIME(b64 string) string {
	if len(b64) < 4 {
		return "image/png"
	}
	// 常见图片格式的 Base64 头部特征
	switch {
	case strings.HasPrefix(b64, "/9j/"):
		return "image/jpeg"
	case strings.HasPrefix(b64, "iVBOR"):
		return "image/png"
	case strings.HasPrefix(b64, "R0lG"):
		return "image/gif"
	case strings.HasPrefix(b64, "UklGR"):
		return "image/webp"
	case strings.HasPrefix(b64, "SUk"):
		return "image/tiff"
	case strings.HasPrefix(b64, "Qk"):
		return "image/bmp"
	case strings.HasPrefix(b64, "PHI"):
		return "image/vnd.adobe.photoshop"
	default:
		return "image/png"
	}
}

// ==================== 视觉代理模型（Vision Proxy） ====================
// 当主模型不支持图片（Capabilities 无 vision）但配置了 VisionProxyModel 时，
// 先用视觉代理模型识别图片内容，再把识别文本合并进主模型请求，图片本身不再转发给主模型。
// 识别结果按「图片内容 + 提示词」哈希缓存到本地 vision_cache/ 目录，重复图片直接命中缓存，不浪费 token。

// visionCacheDir 视觉识别结果缓存目录
const visionCacheDir = "vision_cache"

// defaultVisionProxyPrompt 默认视觉代理提示词
const defaultVisionProxyPrompt = "请仔细查看这张图片，用简体中文详细描述图片中的全部内容。包括：1) 图片中的文字（OCR，逐字提取）；2) 图片中的物体、场景、人物；3) 图表、表格、代码等结构化内容请完整还原。描述要详尽准确，不要遗漏任何细节。"

// visionCacheMutex 保护视觉缓存目录的并发读写
var visionCacheMutex sync.Mutex

// visionImage 表示请求中的一张图片（base64 数据 + MIME 类型）
type visionImage struct {
	Base64 string
	MIME   string
}

// visionCachePath 计算图片识别结果的缓存文件路径
// 缓存键 = SHA256(图片base64 + 提示词)，命中缓存则直接复用识别结果，不重复调用代理模型
func visionCachePath(img visionImage, prompt string) string {
	sum := sha256.Sum256([]byte(img.Base64 + "\x00" + prompt))
	return filepath.Join(visionCacheDir, hex.EncodeToString(sum[:])+".txt")
}

// loadVisionCache 读取视觉识别缓存，命中返回 true 和缓存内容
func loadVisionCache(img visionImage, prompt string) (string, bool) {
	visionCacheMutex.Lock()
	defer visionCacheMutex.Unlock()
	data, err := os.ReadFile(visionCachePath(img, prompt))
	if err != nil {
		return "", false
	}
	return string(data), true
}

// saveVisionCache 保存视觉识别结果到本地缓存
func saveVisionCache(img visionImage, prompt string, result string) {
	visionCacheMutex.Lock()
	defer visionCacheMutex.Unlock()
	if err := os.MkdirAll(visionCacheDir, 0755); err != nil {
		return
	}
	_ = os.WriteFile(visionCachePath(img, prompt), []byte(result), 0644)
}

// recognizeImageWithProxy 调用视觉代理模型识别单张图片，返回识别文本
// 优先命中本地缓存；未命中则调用上游视觉模型，成功后写入缓存
func recognizeImageWithProxy(img visionImage, proxyModel string, prompt string) (string, error) {
	// 1. 尝试命中本地缓存
	if cached, ok := loadVisionCache(img, prompt); ok {
		fmt.Println("💾 视觉代理: 命中本地缓存，跳过识别")
		return cached, nil
	}

	// 2. 构建识别请求（只发图片 + 提示词，不携带历史消息，节省 token）
	userContent := []interface{}{
		map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]interface{}{
				"url": "data:" + img.MIME + ";base64," + img.Base64,
			},
		},
		map[string]interface{}{
			"type": "text",
			"text": prompt,
		},
	}
	payload := map[string]interface{}{
		"model": proxyModel,
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": userContent},
		},
		"stream": false,
	}
	b, _ := json.Marshal(payload)

	httpReq, err := http.NewRequest("POST", cfg.OpenAIBase+"/chat/completions", bytes.NewBuffer(b))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("视觉代理模型返回 HTTP %d: %s", resp.StatusCode, string(raw))
	}

	// 3. 解析识别结果（支持 content 为 string 或数组两种格式）
	var upstreamResp struct {
		Choices []struct {
			Message struct {
				Content interface{} `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &upstreamResp); err != nil {
		return "", err
	}
	if len(upstreamResp.Choices) == 0 {
		return "", errors.New("视觉代理模型无返回内容")
	}

	var result string
	switch c := upstreamResp.Choices[0].Message.Content.(type) {
	case string:
		result = c
	case []interface{}:
		var parts []string
		for _, block := range c {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if t, _ := blockMap["type"].(string); t == "text" {
					if text, ok := blockMap["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		result = strings.Join(parts, "\n")
	default:
		result = fmt.Sprintf("%v", c)
	}
	result = strings.TrimSpace(result)
	if result == "" {
		return "", errors.New("视觉代理模型返回空内容")
	}

	// 4. 保存缓存
	saveVisionCache(img, prompt, result)
	return result, nil
}

// getVisionProxySetting 获取指定模型的视觉代理配置（代理模型ID + 提示词）
// 返回 (代理模型ID, 提示词, 是否启用)
// 提示词优先级：模型自定义 VisionProxyPrompt > 全局 VisionProxyPrompt > 内置默认提示词
func getVisionProxySetting(modelID string) (string, string, bool) {
	setting, ok := cfg.ModelDetailedSettings[modelID]
	if !ok || strings.TrimSpace(setting.VisionProxyModel) == "" {
		return "", "", false
	}
	prompt := strings.TrimSpace(setting.VisionProxyPrompt)
	if prompt == "" {
		prompt = strings.TrimSpace(cfg.VisionProxyPrompt)
	}
	if prompt == "" {
		prompt = defaultVisionProxyPrompt
	}
	return setting.VisionProxyModel, prompt, true
}

// applyVisionProxy 视觉代理主入口：若请求包含图片且主模型配置了视觉代理，
// 则用代理模型识别所有图片，将识别文本合并进请求（图片本身移除），返回处理后的请求体。
// 返回 (新请求体, 是否发生了代理处理)
//
// 重要：识别结果只追加到「包含该图片的那条消息」中，不能全局合并后追加到所有含图片的消息。
// 否则第二次对话时（客户端会携带历史消息），历史图片消息会被追加新图片的描述，
// 导致 AI 混淆不同图片（例如把图片 B 误认为图片 A）。
func applyVisionProxy(body []byte, modelID string) ([]byte, bool) {
	proxyModel, prompt, ok := getVisionProxySetting(modelID)
	if !ok {
		return body, false
	}

	// 解析请求体
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, false
	}

	messages, _ := req["messages"].([]interface{})
	modified := false

	for _, msg := range messages {
		msgMap, _ := msg.(map[string]interface{})
		if msgMap == nil {
			continue
		}

		// 提取本条消息中的图片（支持四种格式）
		var msgImages []visionImage
		hasImagesArray := false
		hasImageBlock := false

		// 格式1: Ollama images 数组
		if imagesRaw, ok := msgMap["images"].([]interface{}); ok && len(imagesRaw) > 0 {
			hasImagesArray = true
			for _, img := range imagesRaw {
				if imgStr, ok := img.(string); ok && imgStr != "" {
					msgImages = append(msgImages, visionImage{Base64: imgStr, MIME: detectImageMIME(imgStr)})
				}
			}
		}

		// 格式2/3/4: content 数组中的图片块
		if contentArr, ok := msgMap["content"].([]interface{}); ok {
			for _, block := range contentArr {
				blockMap, _ := block.(map[string]interface{})
				if blockMap == nil {
					continue
				}
				switch blockMap["type"] {
				case "image":
					// Ollama: {"type":"image","image_base64":"..."}
					if b64, ok := blockMap["image_base64"].(string); ok && b64 != "" {
						msgImages = append(msgImages, visionImage{Base64: b64, MIME: detectImageMIME(b64)})
						hasImageBlock = true
						continue
					}
					// Anthropic: {"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}
					if source, ok := blockMap["source"].(map[string]interface{}); ok {
						data, _ := source["data"].(string)
						if data != "" {
							mediaType, _ := source["media_type"].(string)
							if mediaType == "" {
								mediaType = detectImageMIME(data)
							}
							msgImages = append(msgImages, visionImage{Base64: data, MIME: mediaType})
							hasImageBlock = true
						}
					}
				case "image_url":
					// OpenAI: {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
					if iu, ok := blockMap["image_url"].(map[string]interface{}); ok {
						urlStr, _ := iu["url"].(string)
						if urlStr != "" {
							if idx := strings.Index(urlStr, ";base64,"); idx >= 0 {
								mime := strings.TrimPrefix(urlStr[:idx], "data:")
								msgImages = append(msgImages, visionImage{Base64: urlStr[idx+len(";base64,"):], MIME: mime})
								hasImageBlock = true
							}
						}
					}
				}
			}
		}

		if len(msgImages) == 0 {
			continue
		}

		// 统一处理：每条含图片的消息都「缓存优先，未命中则识别」。
		// 注意：不能只识别最后一条 user 消息的图片——图片可能出现在 system 消息、
		// 或非最后一条 user 消息中（先发图 → AI 回复 → 文字追问），跳过会导致图片不识别。
		// 历史图片命中缓存时直接复用（毫秒级），未命中才调用代理模型。
		var descriptions []string
		var pending []visionImage
		for _, img := range msgImages {
			if desc, ok := loadVisionCache(img, prompt); ok {
				descriptions = append(descriptions, desc)
			} else {
				pending = append(pending, img)
			}
		}
		if len(pending) > 0 {
			fmt.Printf("🖼️ 视觉代理: 本条消息 %d 张图片中 %d 张未命中缓存，并发识别中...\n", len(msgImages), len(pending))
			// 并发识别（限制最多 3 个并发，避免打爆上游限流）
			concurrency := 3
			if len(pending) < concurrency {
				concurrency = len(pending)
			}
			sem := make(chan struct{}, concurrency)
			var wg sync.WaitGroup
			results := make([]string, len(pending))
			for i, img := range pending {
				wg.Add(1)
				sem <- struct{}{}
				go func(idx int, img visionImage) {
					defer wg.Done()
					defer func() { <-sem }()
					desc, err := recognizeImageWithProxy(img, proxyModel, prompt)
					if err != nil {
						fmt.Printf("⚠️ 视觉代理: 第 %d 张图片识别失败: %v\n", idx+1, err)
						return
					}
					results[idx] = desc
				}(i, img)
			}
			wg.Wait()
			for _, desc := range results {
				if desc != "" {
					descriptions = append(descriptions, desc)
				}
			}
		}
		if len(descriptions) == 0 {
			continue
		}

		mergedText := strings.Join(descriptions, "\n\n")

		// 移除 images 数组
		if hasImagesArray {
			delete(msgMap, "images")
			modified = true
		}

		// 移除 content 数组中的图片块，识别文本合并进本条消息
		if hasImageBlock || hasImagesArray {
			switch c := msgMap["content"].(type) {
			case string:
				// content 为纯字符串：识别文本直接追加到字符串末尾
				msgMap["content"] = c + "\n\n[图片内容识别结果]\n" + mergedText
			case []interface{}:
				// content 为数组：移除图片块，识别文本作为 text 块追加
				var newContent []interface{}
				for _, block := range c {
					blockMap, _ := block.(map[string]interface{})
					if blockMap == nil {
						newContent = append(newContent, block)
						continue
					}
					switch blockMap["type"] {
					case "image", "image_url":
						// 图片块移除
					default:
						newContent = append(newContent, block)
					}
				}
				newContent = append(newContent, map[string]interface{}{
					"type": "text",
					"text": "\n\n[图片内容识别结果]\n" + mergedText,
				})
				msgMap["content"] = newContent
			default:
				// content 缺失或为空：直接设置为识别文本
				msgMap["content"] = "[图片内容识别结果]\n" + mergedText
			}
			modified = true
		}
	}

	if !modified {
		return body, false
	}

	newBody, err := json.Marshal(req)
	if err != nil {
		return body, false
	}
	fmt.Println("✅ 视觉代理: 图片已替换为识别文本，请求体已更新")
	return newBody, true
}

// makeOllamaMessage 构建 Ollama 格式的消息响应
func makeOllamaMessage(role string, content string, toolCalls []OllamaToolCall, thinkingContent string) map[string]interface{} {
	msg := map[string]interface{}{
		"role":    role,
		"content": content,
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	if thinkingContent != "" {
		msg["thinking"] = thinkingContent
	}
	return msg
}

func getDefaultConfig() Config {
	return Config{
		IP:                    "0.0.0.0",
		PORT:                  "11434",
		Log_Limit:             100,
		Log_Responses:         true,
		Log_Headers:           true,
		Log_Body:              true,
		OpenAIPrefix:          "[VC反代] ",
		OpenAISuffix:          "",
		StreamMode:            streamModePreserve,
		Capabilities:          []string{"tools", "vision"}, // vs2026 需要这个字段才能启用工具功能
		OpenAIBase:            "https://api.openai.com/v1",
		OpenAIKey:             "",
		ModelAlias:            map[string]string{},               // 模型别名：key=上游模型ID, value=显示名称
		ModelDetailedSettings: map[string]ModelDetailedSetting{}, // 模型详细设置：key=上游模型ID, value={ContextLength, MaxOutputTokens, Capabilities}
		RequestPromptReplace:  map[string]PromptReplaceRule{},    // 请求提示词替换规则
		VisionProxyPrompt:     defaultVisionProxyPrompt,          // 全局默认视觉代理提示词
	}
}

func printConfigHelp() {
	fmt.Println("")
	fmt.Println("══════════════════════════════════════════════ 🪄 配置项说明 ═════════════════════════════════════════════════")
	fmt.Println(" ▼ IP              : 监听地址 (默认 0.0.0.0，本机测试用 127.0.0.1)")
	fmt.Println(" ▼ PORT            : 监听端口 (默认 11434，即 Ollama 默认端口)")
	fmt.Println(" ▼ Log_Limit       : 终端自动清理的日志行数阈值")
	fmt.Println(" ▼ Log_Responses   : 是否打印响应内容 (true/false)")
	fmt.Println(" ▼ Log_Headers     : 是否打印请求头 (true/false)")
	fmt.Println(" ▼ Log_Body        : 是否打印请求体 (true/false)")
	fmt.Println(" ▼ OpenAI_Prefix   : 返回给客户端的模型名称前缀,仅影响模型名字显示")
	fmt.Println(" ▼ OpenAI_Suffix   : 返回给客户端的模型名称后缀,仅影响模型名字显示")
	fmt.Println(" ▼ StreamMode      : 流式策略 = 不覆写/强制流式/强制关闭 (preserve/force_stream/force_close)")
	fmt.Println(" ▼ Capabilities    : 向客户端声明支持的能力列表 (tools, vision 等)")
	fmt.Println(" ▼ OPENAI_BASE     : 上游 OpenAI 兼容 API 地址 (必填)")
	fmt.Println(" ▼ OPENAI_KEY      : 上游 API 密钥 (必填，每次启动时自动加密存储,换设备需重新输入)")
	fmt.Println(" ▼ ModelAlias      : 模型别名映射,仅影响模型名字显示 {上游模型ID: 显示名称, 上游模型ID: 显示名称, ...}")
	fmt.Println(" ▼ ModelDetailedSettings : 模型详细设置,覆盖上游自动获取的值")
	fmt.Println("                     格式: {上游模型ID: {ContextLength: 上下文长度, MaxOutputTokens: 最大输出, Capabilities: [能力列表], VisionProxyModel: 视觉代理模型ID, VisionProxyPrompt: 视觉代理提示词}}")
	fmt.Println("                     当 Capabilities 有定义时,优先使用此处的配置,否则使用全局 Capabilities")
	fmt.Println("                     VisionProxyModel: 主模型不支持图片时,收到图片请求会先用此模型识别图片,")
	fmt.Println("                       再把识别文本合并进主模型请求(图片本身不再转发给主模型)")
	fmt.Println("                     VisionProxyPrompt: 视觉代理自定义提示词(留空=使用全局 VisionProxyPrompt)")
	fmt.Println("                     识别结果按图片内容+提示词哈希缓存到 vision_cache/ 目录,重复图片直接命中缓存,不浪费 token")
	fmt.Println("                     示例: {\"gpt-4o\": {\"ContextLength\": 128000, \"MaxOutputTokens\": 16384}}")
	fmt.Println("                     示例: {\"deepseek-chat\": {\"VisionProxyModel\": \"gpt-4o\", \"VisionProxyPrompt\": \"请描述图片内容\"}}")
	fmt.Println(" ▼ VisionProxyPrompt : 全局默认视觉代理提示词,模型未自定义提示词时使用(留空=内置默认提示词)")
	fmt.Println("                     优先级: 模型自定义 VisionProxyPrompt > 全局 VisionProxyPrompt > 内置默认提示词")
	fmt.Println(" ▼ RequestPromptReplace: 请求提示词替换规则,自动替换请求中的指定文本")
	fmt.Println("                     格式: {规则名称: {enable, role, index, prompt, replace}}")
	fmt.Println("                     优先级:")
	fmt.Println("                       role+index → 先按 role 过滤,再取第 N 条替换")
	fmt.Println("                       role 单独  → 替换所有匹配 role 的消息")
	fmt.Println("                       index 单独 → 按索引取第 N 条替换")
	fmt.Println("                     示例: {\"替换系统提示词\": {\"enable\": true, \"role\": \"system\", \"index\": 0, \"prompt\": \"你是一个AI\", \"replace\": \"你是助手\"}}")
	fmt.Println("════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("")
}

func printModelAliases() {
	// 获取上游模型列表
	req, err := http.NewRequest("GET", cfg.OpenAIBase+"/models", nil)
	if err != nil {
		fmt.Println("⚠️ 无法获取上游模型列表:", err)
		return
	}
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("⚠️ 无法连接上游获取模型列表:", err)
		return
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)

	var upstream struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &upstream); err != nil || len(upstream.Data) == 0 {
		fmt.Println("⚠️ 上游模型列表解析失败")
		return
	}

	// 获取模型元数据（上下文长度、输出上限等）
	modelMeta := fetchUpstreamModelMeta()

	fmt.Println("📋 上游拥有的模型:")
	for _, m := range upstream.Data {
		displayName := m.ID
		if alias, ok := cfg.ModelAlias[m.ID]; ok && alias != "" {
			displayName = alias
		}
		displayName = cfg.OpenAIPrefix + displayName + cfg.OpenAISuffix

		// 获取该模型的上下文信息
		meta := modelMeta[m.ID]
		ctxLen := meta.ContextLength
		maxOut := meta.MaxOutputTokens
		if ctxLen <= 0 {
			ctxLen = DefaultContextLength
		}
		if maxOut <= 0 {
			maxOut = DefaultMaxOutputTokens
		}

		fmt.Printf("   🧩 %s\n", displayName)
		fmt.Printf("       📎 上游模型ID: %s\n", m.ID)
		if alias, ok := cfg.ModelAlias[m.ID]; ok && alias != "" {
			fmt.Printf("       🔖 别名映射:   %s → %s\n", m.ID, alias)
		} else {
			fmt.Printf("       🔖 别名映射:   未设置，使用原始名称\n")
		}
		fmt.Printf("       📐 上下文长度: %d\n", ctxLen)
		fmt.Printf("       📤 最大输出:   %d\n", maxOut)
		// 优先使用 ModelDetailedSettings 中的 Capabilities
		modelCaps := cfg.Capabilities
		if setting, ok := cfg.ModelDetailedSettings[m.ID]; ok && len(setting.Capabilities) > 0 {
			modelCaps = setting.Capabilities
		}
		fmt.Printf("       🛠️  能力集合:   %v\n", modelCaps)
	}
	fmt.Println("")
}

func loadConfig() {
	defaultCfg := getDefaultConfig()

	data, err := os.ReadFile("config.json")
	if err != nil {
		// config.json 不存在 → 自动创建默认配置
		fmt.Println("📝 config.json 不存在，正在自动创建默认配置...")
		if err := saveConfig(defaultCfg); err != nil {
			fmt.Println("❌ 无法创建 config.json:", err)
			pauseAndExit()
		}
		fmt.Println("✅ config.json 已创建，请填写 OPENAI_BASE 和 OPENAI_KEY 后重新启动程序")
		pauseAndExit()
	}

	// 解析 JSON 原始字段，用于检测缺失字段
	var rawMap map[string]interface{}
	if err := json.Unmarshal(data, &rawMap); err != nil {
		fmt.Println("config.json 格式解析失败")
		pauseAndExit()
	}

	var stored Config
	if err := json.Unmarshal(data, &stored); err != nil {
		fmt.Println("config.json 格式解析失败")
		pauseAndExit()
	}

	// 检测每个字段是否存在于 JSON 中，缺失则用默认值补充
	needSave := false
	if _, ok := rawMap["IP"]; !ok {
		stored.IP = defaultCfg.IP
		needSave = true
	}
	if _, ok := rawMap["PORT"]; !ok {
		stored.PORT = defaultCfg.PORT
		needSave = true
	}
	if _, ok := rawMap["Log_Limit"]; !ok {
		stored.Log_Limit = defaultCfg.Log_Limit
		needSave = true
	}
	if _, ok := rawMap["Log_Responses"]; !ok {
		stored.Log_Responses = defaultCfg.Log_Responses
		needSave = true
	}
	if _, ok := rawMap["Log_Headers"]; !ok {
		stored.Log_Headers = defaultCfg.Log_Headers
		needSave = true
	}
	if _, ok := rawMap["Log_Body"]; !ok {
		stored.Log_Body = defaultCfg.Log_Body
		needSave = true
	}
	if _, ok := rawMap["OpenAI_Prefix"]; !ok {
		stored.OpenAIPrefix = defaultCfg.OpenAIPrefix
		needSave = true
	}
	if _, ok := rawMap["OpenAI_Suffix"]; !ok {
		stored.OpenAISuffix = defaultCfg.OpenAISuffix
		needSave = true
	}
	if _, ok := rawMap["StreamMode"]; ok {
		stored.StreamMode = normalizeStreamMode(stored.StreamMode)
		if stored.StreamMode == "" {
			stored.StreamMode = defaultCfg.StreamMode
			needSave = true
		}
	} else if legacyEnabled, ok := rawMap["EnableStream"].(bool); ok {
		stored.StreamMode = streamModeFromLegacy(legacyEnabled)
		needSave = true
	} else {
		stored.StreamMode = defaultCfg.StreamMode
		needSave = true
	}
	if _, ok := rawMap["Capabilities"]; !ok {
		stored.Capabilities = defaultCfg.Capabilities
		needSave = true
	}
	if _, ok := rawMap["OPENAI_BASE"]; !ok {
		stored.OpenAIBase = defaultCfg.OpenAIBase
		needSave = true
	}
	if _, ok := rawMap["OPENAI_KEY"]; !ok {
		stored.OpenAIKey = defaultCfg.OpenAIKey
		needSave = true
	}
	if _, ok := rawMap["ModelAlias"]; !ok {
		stored.ModelAlias = defaultCfg.ModelAlias
		needSave = true
	}
	if _, ok := rawMap["RequestPromptReplace"]; !ok {
		stored.RequestPromptReplace = defaultCfg.RequestPromptReplace
		needSave = true
	}
	if _, ok := rawMap["VisionProxyPrompt"]; !ok {
		stored.VisionProxyPrompt = defaultCfg.VisionProxyPrompt
		needSave = true
	}
	// 迁移旧版替换规则：force / replaceWhole 布尔字段 → mode 枚举
	if migrateRequestPromptReplace(&stored, rawMap) {
		needSave = true
		fmt.Println("🔄 检测到旧的替换规则字段（force/replaceWhole），已迁移为 mode 枚举")
	}

	// 优先读取新字段 ModelDetailedSettings，兼容旧字段 ModelTokenSettings
	if _, ok := rawMap["ModelDetailedSettings"]; ok {
		// 新字段已存在，无需处理
	} else if oldVal, ok := rawMap["ModelTokenSettings"]; ok {
		// 旧字段存在 → 迁移到新字段
		if oldMap, ok := oldVal.(map[string]interface{}); ok {
			migrated := make(map[string]ModelDetailedSetting)
			for k, v := range oldMap {
				if oldSetting, ok := v.(map[string]interface{}); ok {
					newSetting := ModelDetailedSetting{}
					if cl, ok := oldSetting["ContextLength"].(float64); ok {
						newSetting.ContextLength = int64(cl)
					}
					if mot, ok := oldSetting["MaxOutputTokens"].(float64); ok {
						newSetting.MaxOutputTokens = int64(mot)
					}
					migrated[k] = newSetting
				}
			}
			stored.ModelDetailedSettings = migrated
		}
		needSave = true
		fmt.Println("🔄 检测到旧的 ModelTokenSettings，已自动迁移到 ModelDetailedSettings")
	} else {
		stored.ModelDetailedSettings = defaultCfg.ModelDetailedSettings
		needSave = true
	}

	if needSave {
		fmt.Println("🔄 检测到 config.json 有新字段，已自动更新")
		if err := saveConfig(stored); err != nil {
			fmt.Println("❌ 无法更新 config.json:", err)
			pauseAndExit()
		}
	}

	if stored.OpenAIBase == "" || stored.OpenAIKey == "" {
		fmt.Println("config.json 缺少 OPENAI_BASE 或 OPENAI_KEY")
		pauseAndExit()
	}

	stored.StreamMode = normalizeStreamMode(stored.StreamMode)

	plainKey, persistedKey, err := normalizeOpenAIKey(stored.OpenAIKey)
	if err != nil {
		fmt.Println("🔒 OPENAI_KEY 校验失败:", err)
		pauseAndExit()
	}

	// 处理 WebConfigPassword：明文自动加密回写；已加密则解密供登录比对
	var plainPw, persistedPw string
	if stored.WebConfigPassword != "" {
		plainPw, persistedPw, err = normalizeWebConfigPassword(stored.WebConfigPassword)
		if err != nil {
			fmt.Println("🔒 WebConfigPassword 校验失败:", err)
			pauseAndExit()
		}
	}

	cfg = stored
	cfg.OpenAIKey = plainKey
	cfg.WebConfigPassword = plainPw

	if persistedKey != "" || persistedPw != "" {
		if persistedKey != "" {
			stored.OpenAIKey = persistedKey
		}
		if persistedPw != "" {
			stored.WebConfigPassword = persistedPw
		}
		if err := saveConfig(stored); err != nil {
			fmt.Println("🔒 配置加密回写失败:", err)
			pauseAndExit()
		}
		if persistedKey != "" {
			fmt.Println("🔒 OPENAI_KEY 已按本机信息加密并回写到 config.json")
		}
		if persistedPw != "" {
			fmt.Println("🔒 WebConfigPassword 已按本机信息加密并回写到 config.json")
		}
	}

	// 记录文件中持久化的密钥/密码格式，供配置管理页面在未修改时原样写回
	storedOpenAIKey = stored.OpenAIKey
	storedWebConfigPassword = stored.WebConfigPassword
}

func pauseAndExit() {
	fmt.Println("按回车键退出...")
	fmt.Scanln()
	os.Exit(1)
}

func normalizeOpenAIKey(value string) (string, string, error) {
	if strings.HasPrefix(value, encryptedKeyPrefix) {
		plainKey, err := decryptOpenAIKey(value)
		return plainKey, "", err
	}

	encryptedKey, err := encryptOpenAIKey(value)
	if err != nil {
		return "", "", err
	}

	return value, encryptedKey, nil
}

// normalizeWebConfigPassword 处理网页访问密码：
// 已加密 → 解密返回明文；明文 → 加密返回持久化值（兼容旧版明文配置）
func normalizeWebConfigPassword(value string) (string, string, error) {
	if value == "" {
		return "", "", nil
	}
	if strings.HasPrefix(value, encryptedKeyPrefix) {
		plain, err := decryptWebConfigPassword(value)
		return plain, "", err
	}
	encrypted, err := encryptWebConfigPassword(value)
	if err != nil {
		return "", "", err
	}
	return value, encrypted, nil
}

// encryptWebConfigPassword 使用与 OPENAI_KEY 相同的 AES-GCM 加密方式（机器指纹+UUID 派生密钥）
func encryptWebConfigPassword(plain string) (string, error) {
	return encryptOpenAIKey(plain)
}

// decryptWebConfigPassword 解密网页访问密码（与 OPENAI_KEY 相同算法）
func decryptWebConfigPassword(value string) (string, error) {
	return decryptOpenAIKey(value)
}

func normalizeStreamMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case streamModeForceStream:
		return streamModeForceStream
	case streamModeForceClose:
		return streamModeForceClose
	case streamModePreserve:
		return streamModePreserve
	default:
		return streamModePreserve
	}
}

func streamModeFromLegacy(enabled bool) string {
	if enabled {
		return streamModePreserve
	}
	return streamModeForceClose
}

func encryptOpenAIKey(plainKey string) (string, error) {
	fingerprint, err := getMachineFingerprint()
	if err != nil {
		return "", err
	}

	block, err := aes.NewCipher(deriveKey(fingerprint, secretUUID))
	if err != nil {
		return "", err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return "", err
	}

	cipherText := gcm.Seal(nil, nonce, []byte(plainKey), []byte(fingerprint))
	payload := append(nonce, cipherText...)

	return encryptedKeyPrefix + fingerprint + "|" + base64.StdEncoding.EncodeToString(payload), nil
}

func decryptOpenAIKey(value string) (string, error) {
	parts := strings.SplitN(value, "|", 3)
	if len(parts) != 3 {
		// 检测是否为旧格式（4段，含 UUID 尾巴）
		oldParts := strings.SplitN(value, "|", 4)
		if len(oldParts) == 4 {
			return "", errors.New("🔒 检测到旧版加密格式已不再支持。请在 config.json 中将 OPENAI_KEY 改为明文密钥后重新启动（程序会自动以新格式加密保存）")
		}
		return "", errors.New("🔒 已加密格式不正确，请在 config.json 中填写明文 OPENAI_KEY 后重新启动（程序会自动加密保存）")
	}

	fingerprint, err := getMachineFingerprint()
	if err != nil {
		return "", err
	}
	if parts[1] != fingerprint {
		return "", errors.New("🔒 机器码不匹配，需要重新输入 OPENAI_KEY")
	}

	// parts[2] 如果含有 |，说明是旧格式（Base64数据|UUID），截断取前半部分
	dataPart := parts[2]
	if idx := strings.Index(dataPart, "|"); idx >= 0 {
		return "", errors.New("🔒 检测到旧版加密格式，已不再支持。请在 config.json 中将 OPENAI_KEY 改为明文密钥后重新启动（程序会自动以新格式加密保存）")
	}

	payload, err := base64.StdEncoding.DecodeString(dataPart)
	if err != nil {
		return "", err
	}

	block, err := aes.NewCipher(deriveKey(fingerprint, secretUUID))
	if err != nil {
		return "", err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}

	nonceSize := gcm.NonceSize()
	if len(payload) < nonceSize {
		return "", errors.New("🔒 已加密数据损坏")
	}

	plainKey, err := gcm.Open(nil, payload[:nonceSize], payload[nonceSize:], []byte(fingerprint))
	if err != nil {
		return "", err
	}

	return string(plainKey), nil
}

func deriveKey(fingerprint, uuid string) []byte {
	sum := sha256.Sum256([]byte("Remote Convert Ollama:" + fingerprint + ":" + uuid))
	return sum[:]
}

func getSystemDriveSerial() (string, error) {
	if runtime.GOOS == "windows" {
		kernel32 := syscall.NewLazyDLL("kernel32.dll")
		getVolumeInfo := kernel32.NewProc("GetVolumeInformationW")

		rootPath, _ := syscall.UTF16PtrFromString("C:\\")
		var volumeSerial uint32

		ret, _, err := getVolumeInfo.Call(
			uintptr(unsafe.Pointer(rootPath)),      // 根目录路径
			0,                                      // 卷名缓冲区（不需要）
			0,                                      // 卷名缓冲区大小
			uintptr(unsafe.Pointer(&volumeSerial)), // ← 序列号输出
			0,                                      // 最大组件长度（不需要）
			0,                                      // 文件系统标志（不需要）
			0,                                      // 文件系统名缓冲区（不需要）
			0,                                      // 文件系统名缓冲区大小
		)
		if ret == 0 {
			return "", fmt.Errorf("获取卷序列号失败: %v", err)
		}

		return fmt.Sprintf("%08X", volumeSerial), nil
	}

	// Linux/macOS 回退：读取 /etc/machine-id
	data, err := os.ReadFile("/etc/machine-id")
	if err == nil {
		return strings.TrimSpace(string(data)), nil
	}

	return "", errors.New("不支持获取系统盘特征码的当前平台")
}

func getMachineFingerprint() (string, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return "", err
	}

	diskSerial, err := getSystemDriveSerial()
	if err != nil {
		return "", err
	}

	parts := []string{
		strings.ToLower(strings.TrimSpace(hostname)),
		runtime.GOOS,
		runtime.GOARCH,
		diskSerial,
	}

	sum := sha256.Sum256([]byte(strings.Join(parts, "|")))
	return hex.EncodeToString(sum[:]), nil
}

func saveConfig(stored Config) error {
	data, err := json.MarshalIndent(stored, "", "\t")
	if err != nil {
		return err
	}

	data = append(data, '\n')
	return os.WriteFile("config.json", data, 0644)
}

// ==================== 本地配置管理网页 ====================

//go:embed web/config.html
var webConfigPageHTML string

// handleConfigPage 返回可视化配置管理页面
func handleConfigPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Write([]byte(webConfigPageHTML))
}

// handleConfigAPI 处理配置的获取与保存
func handleConfigAPI(w http.ResponseWriter, r *http.Request) {
	if !requireConfigAuth(w, r) {
		return
	}
	switch r.Method {
	case http.MethodGet:
		apiGetConfig(w, r)
	case http.MethodPost:
		apiSaveConfig(w, r)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// requireConfigAuth 校验配置管理接口的访问权限。
// 未启用密码（WebConfigPassword 为空）时直接放行；
// 启用后要求请求头 X-Config-Token 携带有效的服务端会话令牌。
// 令牌由登录接口签发、存于服务端内存 —— 服务端重启后全部会话失效，
// 已登录用户必须重新登录，杜绝"重启后仍可免登录修改"的漏洞。
func requireConfigAuth(w http.ResponseWriter, r *http.Request) bool {
	if cfg.WebConfigPassword == "" {
		return true
	}
	if validConfigSession(r.Header.Get("X-Config-Token")) {
		return true
	}
	http.Error(w, "未授权：会话已失效，请重新解锁", http.StatusUnauthorized)
	return false
}

// configSessions 存储已签发的配置管理会话令牌（token → 过期时间）。
// 仅存内存：服务端重启即全部失效，已登录用户必须重新登录。
var (
	configSessions   = make(map[string]time.Time)
	configSessionsMu sync.Mutex
)

// configSessionTTL 会话有效期（24 小时）
const configSessionTTL = 24 * time.Hour

// newConfigSession 生成并登记一个随机会话令牌
func newConfigSession() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	token := hex.EncodeToString(buf)

	configSessionsMu.Lock()
	defer configSessionsMu.Unlock()
	// 顺带清理过期令牌，防止 map 无限膨胀
	now := time.Now()
	for t, exp := range configSessions {
		if now.After(exp) {
			delete(configSessions, t)
		}
	}
	configSessions[token] = now.Add(configSessionTTL)
	return token, nil
}

// validConfigSession 校验令牌是否有效（存在且未过期）
func validConfigSession(token string) bool {
	if token == "" {
		return false
	}
	configSessionsMu.Lock()
	defer configSessionsMu.Unlock()
	expire, ok := configSessions[token]
	if !ok {
		return false
	}
	if time.Now().After(expire) {
		delete(configSessions, token)
		return false
	}
	return true
}

// removeConfigSession 使指定令牌立即失效（退出登录 / 密码变更后）
func removeConfigSession(token string) {
	configSessionsMu.Lock()
	delete(configSessions, token)
	configSessionsMu.Unlock()
}

// clearAllConfigSessions 清除所有会话（修改访问密码后强制全部重新登录）
func clearAllConfigSessions() {
	configSessionsMu.Lock()
	configSessions = make(map[string]time.Time)
	configSessionsMu.Unlock()
}

// apiConfigAuthStatus 返回是否启用了访问密码（公开接口，供登录页判断）
func apiConfigAuthStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"required": cfg.WebConfigPassword != "",
	})
}

// apiConfigLogin 验证访问密码并签发会话令牌（公开接口）
func apiConfigLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, _ := io.ReadAll(r.Body)
	var req struct {
		Password string `json:"password"`
	}
	_ = json.Unmarshal(body, &req)

	if cfg.WebConfigPassword == "" {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": true, "message": "未启用密码保护"})
		return
	}
	if req.Password != cfg.WebConfigPassword {
		http.Error(w, "访问密码错误", http.StatusUnauthorized)
		return
	}
	token, err := newConfigSession()
	if err != nil {
		http.Error(w, "会话创建失败", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":    true,
		"token": token,
	})
}

// apiConfigLogout 使当前会话令牌立即失效（退出登录）
func apiConfigLogout(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	removeConfigSession(r.Header.Get("X-Config-Token"))
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
}

// apiGetConfig 返回当前配置（OPENAI_KEY / WebConfigPassword 脱敏，绝不返回明文）
func apiGetConfig(w http.ResponseWriter, r *http.Request) {
	cfgCopy := cfg
	keySet := cfgCopy.OpenAIKey != ""
	cfgCopy.OpenAIKey = ""
	pwSet := cfgCopy.WebConfigPassword != ""
	cfgCopy.WebConfigPassword = ""
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	json.NewEncoder(w).Encode(struct {
		Config
		OpenAIKeySet         bool `json:"OPENAI_KEY_SET"`
		WebConfigPasswordSet bool `json:"WEB_CONFIG_PASSWORD_SET"`
	}{Config: cfgCopy, OpenAIKeySet: keySet, WebConfigPasswordSet: pwSet})
}

// apiSaveConfig 保存配置到 config.json 并同步更新运行内存
func apiSaveConfig(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "读取请求失败", http.StatusBadRequest)
		return
	}

	var newCfg Config
	if err := json.Unmarshal(body, &newCfg); err != nil {
		http.Error(w, "配置格式解析失败: "+err.Error(), http.StatusBadRequest)
		return
	}

	// 处理 OPENAI_KEY：空=保持不变；已加密=验证后原样保存；明文=自动加密保存
	plainKey := cfg.OpenAIKey // 默认沿用当前内存明文
	switch {
	case newCfg.OpenAIKey == "":
		if storedOpenAIKey == "" {
			http.Error(w, "OPENAI_KEY 不能为空，请填写密钥", http.StatusBadRequest)
			return
		}
		newCfg.OpenAIKey = storedOpenAIKey
	case strings.HasPrefix(newCfg.OpenAIKey, encryptedKeyPrefix):
		pk, err := decryptOpenAIKey(newCfg.OpenAIKey)
		if err != nil {
			http.Error(w, "OPENAI_KEY 校验失败: "+err.Error(), http.StatusBadRequest)
			return
		}
		plainKey = pk
		// 保持用户提供的加密值原样保存
	default:
		pk, persisted, err := normalizeOpenAIKey(newCfg.OpenAIKey)
		if err != nil {
			http.Error(w, "OPENAI_KEY 校验失败: "+err.Error(), http.StatusBadRequest)
			return
		}
		plainKey = pk
		newCfg.OpenAIKey = persisted
	}

	// 校验与规范化
	if strings.TrimSpace(newCfg.OpenAIBase) == "" {
		http.Error(w, "OPENAI_BASE 不能为空", http.StatusBadRequest)
		return
	}
	newCfg.OpenAIBase = strings.TrimSpace(newCfg.OpenAIBase)
	newCfg.IP = strings.TrimSpace(newCfg.IP)
	newCfg.PORT = strings.TrimSpace(newCfg.PORT)
	newCfg.StreamMode = normalizeStreamMode(newCfg.StreamMode)

	// 规范化替换规则模式：mode 枚举 + 兼容旧 force/replaceWhole 字段
	var rawCfg map[string]interface{}
	_ = json.Unmarshal(body, &rawCfg)
	migrateRequestPromptReplace(&newCfg, rawCfg)

	// 处理 WebConfigPassword：空=保持原加密值；明文=自动加密保存；已加密=校验后保存
	pwChanged := false
	if newCfg.WebConfigPassword != "" {
		plainPw, persistedPw, err := normalizeWebConfigPassword(strings.TrimSpace(newCfg.WebConfigPassword))
		if err != nil {
			http.Error(w, "访问密码处理失败: "+err.Error(), http.StatusBadRequest)
			return
		}
		if plainPw != cfg.WebConfigPassword {
			pwChanged = true // 密码确实变更
		}
		cfg.WebConfigPassword = plainPw // 内存明文，供登录比对
		storedWebConfigPassword = persistedPw
		newCfg.WebConfigPassword = persistedPw
	} else {
		newCfg.WebConfigPassword = storedWebConfigPassword
	}

	if err := saveConfig(newCfg); err != nil {
		http.Error(w, "写入 config.json 失败: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// 更新运行内存（map 整体替换引用，避免并发读写同一 map）
	storedOpenAIKey = newCfg.OpenAIKey
	applyConfigToRuntime(newCfg)
	cfg.OpenAIKey = plainKey

	// 访问密码已变更 → 清除所有已签发会话，强制所有用户重新登录
	if pwChanged {
		clearAllConfigSessions()
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]string{
		"ok":      "true",
		"message": "✅ 配置已保存。IP/PORT 修改需重启程序生效，其余配置立即生效。",
	})
}

// applyConfigToRuntime 将新配置应用到运行时的全局 cfg（不含 OpenAIKey / WebConfigPassword，
// 两者由调用方单独处理为明文，避免加密值覆盖内存明文）
func applyConfigToRuntime(newCfg Config) {
	cfg.IP = newCfg.IP
	cfg.PORT = newCfg.PORT
	cfg.Log_Limit = newCfg.Log_Limit
	cfg.Log_Responses = newCfg.Log_Responses
	cfg.Log_Headers = newCfg.Log_Headers
	cfg.Log_Body = newCfg.Log_Body
	cfg.OpenAIPrefix = newCfg.OpenAIPrefix
	cfg.OpenAISuffix = newCfg.OpenAISuffix
	cfg.StreamMode = newCfg.StreamMode
	cfg.Capabilities = newCfg.Capabilities
	cfg.OpenAIBase = newCfg.OpenAIBase
	cfg.ModelAlias = newCfg.ModelAlias
	cfg.ModelDetailedSettings = newCfg.ModelDetailedSettings
	cfg.RequestPromptReplace = newCfg.RequestPromptReplace
	cfg.VisionProxyPrompt = newCfg.VisionProxyPrompt
}

// apiTestConfig 测试上游 API 连通性并返回模型列表
func apiTestConfig(w http.ResponseWriter, r *http.Request) {
	if !requireConfigAuth(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, _ := io.ReadAll(r.Body)
	var req struct {
		OpenAIBase string `json:"OPENAI_BASE"`
		OpenAIKey  string `json:"OPENAI_KEY"`
	}
	_ = json.Unmarshal(body, &req)

	base := strings.TrimSpace(req.OpenAIBase)
	if base == "" {
		base = cfg.OpenAIBase
	}
	key := req.OpenAIKey
	if key == "" {
		key = cfg.OpenAIKey // 内存中为明文
	} else if strings.HasPrefix(key, encryptedKeyPrefix) {
		pk, err := decryptOpenAIKey(key)
		if err != nil {
			http.Error(w, "密钥解密失败: "+err.Error(), http.StatusBadRequest)
			return
		}
		key = pk
	}

	httpReq, err := http.NewRequest("GET", strings.TrimRight(base, "/")+"/models", nil)
	if err != nil {
		http.Error(w, "请求构造失败: "+err.Error(), http.StatusBadRequest)
		return
	}
	httpReq.Header.Set("Authorization", "Bearer "+key)
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		http.Error(w, "连接失败: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		http.Error(w, fmt.Sprintf("上游返回错误 (%d): %s", resp.StatusCode, truncateStr(string(raw), 300)), http.StatusBadGateway)
		return
	}

	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	_ = json.Unmarshal(raw, &models)

	ids := make([]string, 0, len(models.Data))
	for _, m := range models.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ok":     true,
		"models": ids,
	})
}

// truncateStr 截断过长的错误信息
func truncateStr(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "..."
}

// upstreamModelMeta 上游模型元数据
type upstreamModelMeta struct {
	ContextLength   int64
	MaxOutputTokens int64
}

// fetchUpstreamModelMeta 调用上游 /v1/models 获取模型元数据并构建映射，
// 然后合并 ModelDetailedSettings 中手动指定的值（手动设置优先）
func fetchUpstreamModelMeta() map[string]upstreamModelMeta {
	result := make(map[string]upstreamModelMeta)

	req, err := http.NewRequest("GET", cfg.OpenAIBase+"/models", nil)
	if err != nil {
		return applyManualModelSettings(result)
	}
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return applyManualModelSettings(result)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)

	// 先用通用 map 解析，捕获所有可能字段
	var upstreamResp struct {
		Data []map[string]interface{} `json:"data"`
	}
	if err := json.Unmarshal(raw, &upstreamResp); err != nil {
		return applyManualModelSettings(result)
	}

	for _, modelData := range upstreamResp.Data {
		id, _ := modelData["id"].(string)
		if id == "" {
			continue
		}

		info := upstreamModelMeta{
			ContextLength:   DefaultContextLength,
			MaxOutputTokens: DefaultMaxOutputTokens,
		}

		// 尝试从多种可能的字段名中提取上下文长度
		for _, field := range []string{"context_length", "max_input_tokens", "max_input_length", "context_window", "max_context"} {
			if v := extractNumeric(modelData, field); v > 0 {
				info.ContextLength = v
				break
			}
		}

		// 尝试提取最大输出 token 数
		for _, field := range []string{"max_output_tokens", "max_completion_tokens", "max_tokens"} {
			if v := extractNumeric(modelData, field); v > 0 {
				info.MaxOutputTokens = v
				break
			}
		}

		result[id] = info
	}

	// 合并 ModelDetailedSettings 手动配置（手动设置优先）
	return applyManualModelSettings(result)
}

// applyManualModelSettings 将 ModelDetailedSettings 中的手动配置合并到 result 中
func applyManualModelSettings(result map[string]upstreamModelMeta) map[string]upstreamModelMeta {
	for modelID, setting := range cfg.ModelDetailedSettings {
		if setting.ContextLength <= 0 && setting.MaxOutputTokens <= 0 {
			continue
		}
		info := result[modelID]
		if setting.ContextLength > 0 {
			info.ContextLength = setting.ContextLength
		}
		if setting.MaxOutputTokens > 0 {
			info.MaxOutputTokens = setting.MaxOutputTokens
		}
		result[modelID] = info
	}
	return result
}

// extractNumeric 从 map 中提取 int64 数值字段（支持 float64 / int / int64 / json.Number）
func extractNumeric(data map[string]interface{}, field string) int64 {
	v, ok := data[field]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case float64:
		return int64(n)
	case int64:
		return n
	case int:
		return int64(n)
	case json.Number:
		val, err := n.Int64()
		if err == nil {
			return val
		}
	}
	return 0
}

// mapFinishReason 将 OpenAI 的 finish_reason 映射为 Ollama 的 done_reason
func mapFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "stop"
	case "length":
		return "length"
	case "tool_calls":
		return "stop" // Ollama 无独立 tool_calls 类型，归为 stop
	case "content_filter":
		return "stop"
	default:
		return "stop" // 未知原因默认 stop
	}
}

// hasCapability 检查 capabilities 列表中是否包含指定的能力
func hasCapability(caps []string, capability string) bool {
	for _, c := range caps {
		if c == capability {
			return true
		}
	}
	return false
}

// estimateTokens 估算文本的 token 数（粗略估计：ASCII 约 4 字符=1 token，中文等非 ASCII 约 1 字=1 token）
// 用途：上游不返回 usage 时本地估算，保证 VS Code 上下文占用显示不为 0
func estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	ascii := 0
	nonASCII := 0
	for _, r := range text {
		if r < 128 {
			ascii++
		} else {
			nonASCII++
		}
	}
	tokens := ascii/4 + nonASCII
	if tokens < 1 {
		tokens = 1
	}
	return tokens
}

// estimatePromptTokens 估算消息列表的 prompt token 数（含角色/结构开销，图片按固定值粗估）
func estimatePromptTokens(messages []interface{}) int {
	total := 0
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		total += 4 // role 等结构开销
		switch c := msgMap["content"].(type) {
		case string:
			total += estimateTokens(c)
		case []interface{}:
			// content 数组（图片/文本块）
			for _, block := range c {
				if bm, ok := block.(map[string]interface{}); ok {
					if t, ok := bm["text"].(string); ok {
						total += estimateTokens(t)
					} else if _, isImg := bm["image_url"]; isImg {
						total += 800 // 图片粗略估算
					} else if _, isImg2 := bm["image"]; isImg2 {
						total += 800
					}
				}
			}
		}
		// tool_calls 参数也计入
		if tcs, ok := msgMap["tool_calls"].([]interface{}); ok {
			for _, tc := range tcs {
				if tm, ok := tc.(map[string]interface{}); ok {
					if fn, ok := tm["function"].(map[string]interface{}); ok {
						if args, ok := fn["arguments"].(string); ok {
							total += estimateTokens(args)
						}
					}
				}
			}
		}
	}
	return total
}

// -------------------- Ollama API: /api/chat --------------------
func ollamaChat(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	var req map[string]interface{}
	json.Unmarshal(body, &req)

	model := "deepseek-chat"
	if m, ok := req["model"].(string); ok {
		model = m
	}

	var messages []interface{}
	if rawMessages, ok := req["messages"].([]interface{}); ok {
		// 将 Ollama 格式的消息转为 OpenAI 格式
		messages = convertMessagesToOpenAI(rawMessages)
	} else if prompt, ok := req["prompt"].(string); ok {
		messages = []interface{}{
			map[string]interface{}{"role": "user", "content": prompt},
		}
	} else {
		messages = []interface{}{}
	}

	payload := map[string]interface{}{
		"model":    model,
		"messages": messages,
	}

	// 透传 Ollama 请求中的参数到 OpenAI 格式
	for _, key := range []string{"tools", "tool_choice", "temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty", "seed"} {
		if val, ok := req[key]; ok {
			payload[key] = val
		}
	}

	requestedStream, _ := req["stream"].(bool)
	switch normalizeStreamMode(cfg.StreamMode) {
	case streamModeForceStream:
		ollamaChatStream(w, payload)
		return
	case streamModeForceClose:
		// 强制关闭流式，直接走非流式分支
	default:
		if requestedStream {
			ollamaChatStream(w, payload)
			return
		}
	}

	b, _ := json.Marshal(payload)

	httpReq, _ := http.NewRequest("POST", cfg.OpenAIBase+"/chat/completions", bytes.NewBuffer(b))
	httpReq.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		http.Error(w, "upstream error", 500)
		return
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	fmt.Println("UPSTREAM:", string(raw))

	// 解析上游响应
	var upstreamResp map[string]interface{}
	if err := json.Unmarshal(raw, &upstreamResp); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.Write(raw)
		return
	}

	// 提取 choices[0].message
	choices, _ := upstreamResp["choices"].([]interface{})
	content := ""
	var toolCalls []OllamaToolCall
	reasoningContent := ""
	finishReason := ""

	if len(choices) > 0 {
		choice, _ := choices[0].(map[string]interface{})
		// 提取 finish_reason
		if fr, ok := choice["finish_reason"].(string); ok {
			finishReason = fr
		}
		if msg, ok := choice["message"].(map[string]interface{}); ok {
			// 提取 content
			if c, ok := msg["content"].(string); ok {
				content = c
			}
			// 提取 tool_calls
			toolCalls = extractToolCalls(msg)
			// 提取 reasoning_content（DeepSeek 思考模式）
			if rc, ok := msg["reasoning_content"].(string); ok {
				reasoningContent = rc
			}
		}
	}

	// 保存 reasoning_content，供后续请求自动注入
	if reasoningContent != "" {
		setLastReasoningContent(reasoningContent)
	}

	// 提取上游 usage（真实 token 计数）
	inputTokens := int64(0)
	outputTokens := int64(0)
	if usage, ok := upstreamResp["usage"].(map[string]interface{}); ok {
		if pt, ok := usage["prompt_tokens"].(float64); ok && pt > 0 {
			inputTokens = int64(pt)
		}
		if ct, ok := usage["completion_tokens"].(float64); ok && ct > 0 {
			outputTokens = int64(ct)
		}
	}
	// 上游未返回 usage 时本地估算，保证 VS Code 上下文占用显示不为 0
	usageSource := "上游"
	if inputTokens <= 0 {
		inputTokens = int64(estimatePromptTokens(messages))
		usageSource = "估算"
	}
	if outputTokens <= 0 {
		outputTokens = int64(estimateTokens(content) + estimateTokens(reasoningContent))
		usageSource = "估算"
	}
	fmt.Printf("🔢 Token [%s] 输入:%d 输出:%d (finish=%s)\n", usageSource, inputTokens, outputTokens, finishReason)

	out := map[string]interface{}{
		"model":             model,
		"created_at":        time.Now().Format("2006-01-02T15:04:05"),
		"message":           makeOllamaMessage("assistant", content, toolCalls, reasoningContent),
		"done":              true,
		"done_reason":       mapFinishReason(finishReason),
		"total_duration":    1,
		"load_duration":     1,
		"prompt_eval_count": inputTokens,
		"eval_count":        outputTokens,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func ollamaChatStream(w http.ResponseWriter, payload map[string]interface{}) {
	payload["stream"] = true
	b, _ := json.Marshal(payload)

	httpReq, _ := http.NewRequest("POST", cfg.OpenAIBase+"/chat/completions", bytes.NewBuffer(b))
	httpReq.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		http.Error(w, "upstream error", 500)
		return
	}
	defer resp.Body.Close()

	// 检查上游是否返回了错误
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		errMsg := fmt.Sprintf("UPSTREAM ERROR (HTTP %d): %s", resp.StatusCode, string(raw))
		fmt.Println(errMsg)
		// 向客户端返回 Ollama 格式错误
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"model":      payload["model"],
			"created_at": time.Now().Format(time.RFC3339),
			"message":    map[string]string{"role": "assistant", "content": fmt.Sprintf("上游API返回错误: HTTP %d - %s", resp.StatusCode, string(raw))},
			"done":       true,
		})
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	model, _ := payload["model"].(string)
	inputTokens := 0
	outputTokens := 0
	var fullContent strings.Builder
	var reasoningContent strings.Builder // 累积 reasoning_content（思考模式）
	lastThinkingLen := 0                 // 已发送的 thinking 长度（用于增量发送）
	contentStarted := false              // 正文是否已开始输出（之后不再发送 thinking 块，避免思考/正文交错显示混乱）
	reader := bufio.NewReader(resp.Body)

	// 流式 tool_calls 累积器
	type accToolCall struct {
		id      string
		typ     string
		name    string
		argsBld strings.Builder
	}
	var accToolCalls []*accToolCall
	hasToolCalls := false
	upstreamFinishReason := "" // 记录上游返回的 finish_reason

	// 发送 Ollama 流式消息块
	sendOllamaChunk := func(content string, done bool, tokens int, toolCalls []OllamaToolCall, rcFull string) {
		// 只发送 thinking 增量（新追加的部分），避免 Cherry Studio reasoning part 追踪混乱
		rcDelta := ""
		if len(rcFull) > lastThinkingLen {
			rcDelta = rcFull[lastThinkingLen:]
			lastThinkingLen = len(rcFull)
		}
		msg := makeOllamaMessage("assistant", content, toolCalls, rcDelta)
		out := map[string]interface{}{
			"model":      model,
			"created_at": time.Now().Format(time.RFC3339),
			"message":    msg,
			"done":       done,
		}
		if done {
			out["done_reason"] = mapFinishReason(upstreamFinishReason)
			out["total_duration"] = 1
			out["load_duration"] = 1
			out["prompt_eval_count"] = inputTokens
			if outputTokens > 0 {
				out["eval_count"] = outputTokens
			} else {
				out["eval_count"] = tokens
			}
		}
		if err := json.NewEncoder(w).Encode(out); err == nil {
			flusher.Flush()
		}
	}

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			http.Error(w, "stream read error", 500)
			return
		}

		line = strings.TrimSpace(line)
		if line == "" || line == "data: [DONE]" {
			if line == "data: [DONE]" {
				break
			}
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		dataStr := strings.TrimPrefix(line, "data: ")

		// 先用完整 DeltaRaw 解析，保留 tool_calls 等字段
		var rawDelta struct {
			Choices []struct {
				Index        int             `json:"index"`
				Delta        json.RawMessage `json:"delta"`
				FinishReason *string         `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage,omitempty"`
		}
		if err := json.Unmarshal([]byte(dataStr), &rawDelta); err != nil {
			continue
		}

		// 使用标准结构解析
		var chunk OpenAIChunk
		if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
			continue
		}

		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens > 0 {
				inputTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				outputTokens = chunk.Usage.CompletionTokens
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		// 保存上游返回的 finish_reason（用于最终 done_reason）
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			upstreamFinishReason = *choice.FinishReason
		}

		// 检查 delta 中是否包含 tool_calls
		if len(rawDelta.Choices) > 0 && len(rawDelta.Choices[0].Delta) > 0 {
			var deltaMap map[string]interface{}
			if err := json.Unmarshal(rawDelta.Choices[0].Delta, &deltaMap); err == nil {
				// 提取 reasoning_content（DeepSeek 思考模式）
				if rc, ok := deltaMap["reasoning_content"].(string); ok && rc != "" {
					reasoningContent.WriteString(rc)
					if cfg.Log_Responses {
						fmt.Print("[思考:" + rc + "]")
					}
					// 正文开始后，上游可能还会交错输出思考碎片（DeepSeek 思考模式特性）。
					// 此时不再转发 thinking 块，避免客户端出现"思考/正文交错拼接"的混乱显示。
					if !contentStarted {
						// 实时推送 thinking 增量（sendOllamaChunk 内部自动计算增量）
						sendOllamaChunk("", false, 0, nil, reasoningContent.String())
					} else {
						// 正文已开始：同步 lastThinkingLen，防止后续正文块夹带思考碎片
						lastThinkingLen = reasoningContent.Len()
					}
				}
				if tcRaw, ok := deltaMap["tool_calls"]; ok {
					hasToolCalls = true
					if tcList, ok := tcRaw.([]interface{}); ok {
						for _, tc := range tcList {
							if tcMap, ok := tc.(map[string]interface{}); ok {
								idx := 0
								if idxF, ok := tcMap["index"].(float64); ok {
									idx = int(idxF)
								}
								// 扩展累积器数组
								for len(accToolCalls) <= idx {
									accToolCalls = append(accToolCalls, &accToolCall{})
								}
								if accToolCalls[idx] == nil {
									accToolCalls[idx] = &accToolCall{}
								}
								if id, ok := tcMap["id"].(string); ok && id != "" {
									accToolCalls[idx].id = id
								}
								if typ, ok := tcMap["type"].(string); ok && typ != "" {
									accToolCalls[idx].typ = typ
								}
								if funcRaw, ok := tcMap["function"].(map[string]interface{}); ok {
									if name, ok := funcRaw["name"].(string); ok && name != "" {
										accToolCalls[idx].name = name
									}
									if args, ok := funcRaw["arguments"].(string); ok && args != "" {
										accToolCalls[idx].argsBld.WriteString(args)
									}
								}
							}
						}
					}
				}
			}
		}

		// 提取普通文本内容
		content := choice.Delta.Content
		if content != "" {
			fullContent.WriteString(content)
			if cfg.Log_Responses {
				fmt.Print(content)
			}
			contentStarted = true
			sendOllamaChunk(content, false, 0, nil, reasoningContent.String())
		}
	}

	if cfg.Log_Responses {
		if hasToolCalls {
			fmt.Println("\n[Tool Calls Detected]")
		} else {
			fmt.Println("")
		}
		fmt.Println("UPSTREAM STREAM:", fullContent.String())
	}

	// 上游未返回 usage（很多中转流式不返回）时本地估算兜底，
	// 保证 VS Code 上下文占用显示不为 0（VS Code 从 prompt_eval_count/eval_count 计算）
	usageSource := "上游"
	if inputTokens <= 0 {
		if msgs, ok := payload["messages"].([]interface{}); ok {
			inputTokens = estimatePromptTokens(msgs)
		}
		usageSource = "估算"
	}
	if outputTokens <= 0 {
		outputTokens = estimateTokens(fullContent.String()) + estimateTokens(reasoningContent.String())
		usageSource = "估算"
	}
	fmt.Printf("🔢 Token[%s] 输入:%d 输出:%d (finish=%s, tool_calls=%v)\n", usageSource, inputTokens, outputTokens, upstreamFinishReason, hasToolCalls)

	// 构建最终消息
	// 注意：不再要求 finish_reason == "tool_calls" 才下发 tool_calls。
	// 很多上游（qwen3 系列中转、部分网关）流式输出 tool_calls 后 finish_reason 返回
	// "stop" 甚至缺失，旧逻辑会静默丢弃已累积的 tool_calls → 客户端收到空回复 →
	// Agent 认为模型没调用工具而反复重试 → 死循环。
	if hasToolCalls {
		// 将累积的 tool_calls 转为 Ollama 格式
		var ollamaTCs []OllamaToolCall
		tcSeq := 0
		for _, atc := range accToolCalls {
			if atc == nil || (atc.name == "" && atc.argsBld.Len() == 0) {
				continue // 跳过空累积器
			}
			var otc OllamaToolCall
			otc.ID = atc.id // 保留上游 id
			if otc.ID == "" {
				// 上游未给 id 时生成兜底 id（与非流式 extractToolCalls 保持一致），
				// 客户端回传 assistant 消息时可直接与 tool 消息配对
				tcSeq++
				otc.ID = fmt.Sprintf("call_%d", tcSeq)
			}
			otc.Function.Name = atc.name
			// arguments 是 JSON 字符串，转为对象
			argsStr := atc.argsBld.String()
			if argsStr != "" {
				var argsObj interface{}
				if err := json.Unmarshal([]byte(argsStr), &argsObj); err == nil {
					otc.Function.Arguments = argsObj
				} else {
					otc.Function.Arguments = argsStr
				}
			}
			ollamaTCs = append(ollamaTCs, otc)
		}
		// 发送最后一个空内容块标记完成，附带 tool_calls
		sendOllamaChunk("", true, 0, ollamaTCs, reasoningContent.String())
	} else {
		// 普通文本完成
		sendOllamaChunk("", true, outputTokens, nil, reasoningContent.String())
	}

	// 保存 reasoning_content，供后续请求自动注入
	if rc := reasoningContent.String(); rc != "" {
		setLastReasoningContent(rc)
	}
}

// -------------------- OpenAI API: /v1/chat/completions --------------------
func openaiChat(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	// 判断是否为流式请求
	var reqMeta struct {
		Stream bool `json:"stream"`
	}
	json.Unmarshal(body, &reqMeta)

	if reqMeta.Stream {
		openaiChatStream(w, r, body)
		return
	}

	req, _ := http.NewRequest("POST", cfg.OpenAIBase+"/chat/completions", bytes.NewBuffer(body))
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "upstream error", 500)
		return
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	fmt.Println("UPSTREAM:", string(raw))

	// 保存 reasoning_content（DeepSeek 思考模式）并注入 reasoning_text（VS Code 兼容）
	var upstreamResp map[string]interface{}
	if err := json.Unmarshal(raw, &upstreamResp); err == nil {
		if choices, ok := upstreamResp["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if msg, ok := choice["message"].(map[string]interface{}); ok {
					if rc, ok := msg["reasoning_content"].(string); ok && rc != "" {
						setLastReasoningContent(rc)
						// VS Code 兼容：将 reasoning_content 映射为 reasoning_text
						if _, exists := msg["reasoning_text"]; !exists {
							msg["reasoning_text"] = rc
						}
					}
				}
			}
		}
		// 重新序列化（因为可能修改了 message）
		if modified, _ := json.Marshal(upstreamResp); modified != nil {
			raw = modified
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(raw)
}

// 流式响应处理
func openaiChatStream(w http.ResponseWriter, r *http.Request, body []byte) {
	req, _ := http.NewRequest("POST", cfg.OpenAIBase+"/chat/completions", bytes.NewBuffer(body))
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "upstream error", 500)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		contentType := resp.Header.Get("Content-Type")
		if contentType == "" {
			contentType = "application/json"
		}
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(resp.StatusCode)
		w.Write(raw)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	out := func(payload map[string]interface{}) {
		jsonData, err := json.Marshal(payload)
		if err != nil {
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		flusher.Flush()
	}

	reader := bufio.NewReader(resp.Body)
	loggedContent := strings.Builder{}
	thinkingDone := false // 标记 thinking 是否已结束（收到 content 后关闭 reasoning_text 注入）
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
			}
			break
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			if cfg.Log_Responses {
				fmt.Println("UPSTREAM SSE:", line)
			}
			continue
		}

		dataStr := strings.TrimPrefix(line, "data: ")
		if dataStr == "[DONE]" {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			break
		}

		if cfg.Log_Responses {
			fmt.Println("UPSTREAM SSE:", line)
		}

		// 用 RawMessage 解析完整 chunk，保留所有字段（包括 tool_calls）
		var rawChunk struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			Model   string `json:"model"`
			Choices []struct {
				Index        int             `json:"index"`
				Delta        json.RawMessage `json:"delta"`
				FinishReason *string         `json:"finish_reason"`
			} `json:"choices"`
			Usage *json.RawMessage `json:"usage,omitempty"`
		}
		if err := json.Unmarshal([]byte(dataStr), &rawChunk); err != nil {
			continue
		}

		// usage-only chunk 跳过
		if len(rawChunk.Choices) == 0 && rawChunk.Usage != nil {
			if cfg.Log_Responses {
				fmt.Println("UPSTREAM SSE: usage-only chunk skipped")
			}
			continue
		}

		if len(rawChunk.Choices) > 0 {
			choice := rawChunk.Choices[0]
			delta := map[string]interface{}{}

			// 解析 delta 为通用 map，保留全部字段（content, reasoning_content, tool_calls 等）
			if len(choice.Delta) > 0 {
				var deltaMap map[string]interface{}
				if err := json.Unmarshal(choice.Delta, &deltaMap); err == nil {
					delta = deltaMap
					// 检查是否已有 content → 标记 thinking 结束
					if c, ok := deltaMap["content"].(string); ok && c != "" {
						loggedContent.WriteString(c)
						if cfg.Log_Responses {
							fmt.Print(c)
						}
						thinkingDone = true
					}
					// VS Code 兼容：将 reasoning_content 映射为 reasoning_text
					// 只在 thinking 阶段注入，content 开始后停止（避免 VS Code 内部追踪断链）
					if !thinkingDone {
						if rc, ok := delta["reasoning_content"].(string); ok && rc != "" {
							if _, exists := delta["reasoning_text"]; !exists {
								delta["reasoning_text"] = rc
							}
						}
					} else {
						// 正文开始后，上游可能还会交错输出思考碎片（DeepSeek 思考模式特性）。
						// 从 delta 中移除 reasoning_text 和 reasoning_content，
						// 避免客户端出现"思考/正文交错拼接"的混乱显示。
						delete(delta, "reasoning_text")
						delete(delta, "reasoning_content")
					}
				}
			}

			choicePayload := map[string]interface{}{
				"index":         choice.Index,
				"delta":         delta,
				"finish_reason": nil,
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				choicePayload["finish_reason"] = *choice.FinishReason
			}
			payload := map[string]interface{}{
				"id":      rawChunk.ID,
				"object":  rawChunk.Object,
				"created": rawChunk.Created,
				"model":   rawChunk.Model,
				"choices": []map[string]interface{}{choicePayload},
			}
			out(payload)
		}
	}

	if cfg.Log_Responses && loggedContent.Len() > 0 {
		fmt.Println("UPSTREAM STREAM:", loggedContent.String())
	}
}

// -------------------- Anthropic Messages API: /v1/messages --------------------
func anthropicMessages(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	var areq AnthropicReq
	if err := json.Unmarshal(body, &areq); err != nil {
		fmt.Println("ANTHROPIC PARSE ERROR:", err)
		fmt.Println("ANTHROPIC RAW BODY:", string(body))
		http.Error(w, `{"error":{"type":"invalid_request_error","message":"Invalid JSON"}}`, 400)
		return
	}

	if areq.Stream {
		anthropicMessagesStream(w, r, &areq)
		return
	}

	// --- 非流式请求 ---

	// 1. 转换请求体 Anthropic → OpenAI
	openaiBody, err := convertAnthropicToOpenAI(&areq)
	if err != nil {
		http.Error(w, `{"error":{"type":"invalid_request_error","message":"`+err.Error()+`"}}`, 400)
		return
	}

	// 2. 转发到上游
	req, _ := http.NewRequest("POST", cfg.OpenAIBase+"/chat/completions", bytes.NewBuffer(openaiBody))
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, `{"error":{"type":"api_error","message":"upstream error"}}`, 500)
		return
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)

	// 如果上游返回错误，直接透传
	if resp.StatusCode != 200 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(raw)
		return
	}

	// 3. 转换响应体 OpenAI → Anthropic
	anthropicBody, convErr := convertOpenAIToAnthropic(raw, areq.Model)
	if convErr != nil {
		fmt.Println("Anthropic 转换失败:", convErr)
		// 降级：透传原始响应
		w.Header().Set("Content-Type", "application/json")
		w.Write(raw)
		return
	}

	fmt.Printf("UPSTREAM Anthropic: %s\n", anthropicBody)

	w.Header().Set("Content-Type", "application/json")
	w.Write(anthropicBody)
}

// -------------------- /v1/messages/count_tokens --------------------
func anthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	var areq AnthropicReq
	if err := json.Unmarshal(body, &areq); err != nil {
		// 如果解析失败，尝试提取 system 字段后用 interface{} 重试
		fmt.Println("ANTHROPIC COUNT_TOKENS PARSE ERROR:", err)
		http.Error(w, `{"error":{"type":"invalid_request_error","message":"Invalid JSON"}}`, 400)
		return
	}

	// 转换为 OpenAI 格式来估算 token 数
	openaiBody, err := convertAnthropicToOpenAI(&areq)
	if err != nil {
		http.Error(w, `{"error":{"type":"api_error","message":"conversion error"}}`, 500)
		return
	}

	// 发送到上游估算 token 数（max_tokens=1 快速返回）
	var upstreamPayload map[string]interface{}
	json.Unmarshal(openaiBody, &upstreamPayload)
	upstreamPayload["max_tokens"] = 1
	upstreamPayload["stream"] = false
	upstreamBody, _ := json.Marshal(upstreamPayload)

	req, _ := http.NewRequest("POST", cfg.OpenAIBase+"/chat/completions", bytes.NewBuffer(upstreamBody))
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err == nil && resp.StatusCode == 200 {
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		var upstreamResp map[string]interface{}
		if json.Unmarshal(raw, &upstreamResp) == nil {
			if usage, ok := upstreamResp["usage"].(map[string]interface{}); ok {
				var inputTokens int = 0
				if pt, ok := usage["prompt_tokens"].(float64); ok {
					inputTokens = int(pt)
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]interface{}{
					"input_tokens": inputTokens,
				})
				return
			}
		}
	}

	// 降级：基于字符数估算
	textLen := len(string(openaiBody))
	estimatedTokens := textLen / 4
	if estimatedTokens < 1 {
		estimatedTokens = 1
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"input_tokens": estimatedTokens,
	})
}

// --- 流式处理 ---
func anthropicMessagesStream(w http.ResponseWriter, r *http.Request, areq *AnthropicReq) {

	openaiBody, err := convertAnthropicToOpenAI(areq)
	if err != nil {
		http.Error(w, `{"error":{"type":"invalid_request_error","message":"`+err.Error()+`"}}`, 400)
		return
	}

	req, _ := http.NewRequest("POST", cfg.OpenAIBase+"/chat/completions", bytes.NewBuffer(openaiBody))
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, `{"error":{"type":"api_error","message":"upstream error"}}`, 500)
		return
	}
	defer resp.Body.Close()

	// 检查上游 HTTP 状态码
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		errMsg := fmt.Sprintf("UPSTREAM ERROR (HTTP %d): %s", resp.StatusCode, string(raw))
		fmt.Println(errMsg)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(raw)
		return
	}

	// 设置 SSE 响应头
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("x-request-id", generateMsgID())
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	msgID := "msg_" + generateMsgID()
	inputTokens := 0
	outputTokens := 0
	msgStarted := false
	streamClosed := false // 是否已发送 message_stop（防止重复收尾）

	// 内容块跟踪：0=text, 1=tool_use...
	type anthropicBlock struct {
		index       int
		blockType   string // "text" or "tool_use"
		started     bool
		toolUseID   string
		toolUseName string
		accIndex    int // 对应 accToolCalls 累积器下标（-1 = 文本块），用于防重复开块
	}
	var blocks []*anthropicBlock
	currentBlockIndex := 0

	// 流式 tool_calls 累积器（用于 tool_use 块）
	type accToolCall struct {
		id      string
		name    string
		argsBld strings.Builder
	}
	var accToolCalls []*accToolCall

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}

		line = strings.TrimSpace(line)
		if line == "" || line == "data: [DONE]" {
			if line == "data: [DONE]" {
				break
			}
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		dataStr := strings.TrimPrefix(line, "data: ")

		// 用 rawDelta 解析保留所有字段
		var rawDelta struct {
			Choices []struct {
				Index        int             `json:"index"`
				Delta        json.RawMessage `json:"delta"`
				FinishReason *string         `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage,omitempty"`
		}
		if err := json.Unmarshal([]byte(dataStr), &rawDelta); err != nil {
			continue
		}

		// 提取 usage
		if rawDelta.Usage != nil {
			if rawDelta.Usage.PromptTokens > 0 {
				inputTokens = rawDelta.Usage.PromptTokens
			}
			if rawDelta.Usage.CompletionTokens > 0 {
				outputTokens = rawDelta.Usage.CompletionTokens
			}
		}

		if len(rawDelta.Choices) == 0 {
			continue
		}

		choice := rawDelta.Choices[0]
		finishReason := choice.FinishReason

		// 解析 delta 为通用 map
		deltaContent := ""
		var deltaToolCalls []interface{}
		if len(choice.Delta) > 0 {
			var deltaMap map[string]interface{}
			if err := json.Unmarshal(choice.Delta, &deltaMap); err == nil {
				if c, ok := deltaMap["content"].(string); ok {
					deltaContent = c
				}
				if tcRaw, ok := deltaMap["tool_calls"]; ok {
					if tcList, ok := tcRaw.([]interface{}); ok {
						deltaToolCalls = tcList
					}
				}
			}
		}

		// --- 发送 message_start（首次） ---
		if !msgStarted {
			msgStarted = true
			sendSSEEvent(w, flusher, "message_start", map[string]interface{}{
				"type": "message_start",
				"message": map[string]interface{}{
					"id":            msgID,
					"type":          "message",
					"role":          "assistant",
					"content":       []interface{}{},
					"model":         areq.Model,
					"stop_reason":   nil,
					"stop_sequence": nil,
					"usage": map[string]interface{}{
						"input_tokens":  inputTokens,
						"output_tokens": 0,
					},
				},
			})
		}

		// --- 处理 tool_calls delta ---
		for _, tc := range deltaToolCalls {
			tcMap, ok := tc.(map[string]interface{})
			if !ok {
				continue
			}
			idx := 0
			if idxF, ok := tcMap["index"].(float64); ok {
				idx = int(idxF)
			}
			// 扩展累积器
			for len(accToolCalls) <= idx {
				accToolCalls = append(accToolCalls, &accToolCall{})
			}
			if accToolCalls[idx] == nil {
				accToolCalls[idx] = &accToolCall{}
			}
			if id, ok := tcMap["id"].(string); ok && id != "" {
				accToolCalls[idx].id = id
			}
			if funcRaw, ok := tcMap["function"].(map[string]interface{}); ok {
				if name, ok := funcRaw["name"].(string); ok && name != "" {
					accToolCalls[idx].name = name
				}
				if args, ok := funcRaw["arguments"].(string); ok && args != "" {
					accToolCalls[idx].argsBld.WriteString(args)
				}
			}
		}

		// --- 发送适当的 content_block 事件 ---
		// 收尾后抑制：message_stop 已发出，后续 chunk（如 usage-only 残留 delta）
		// 不得再发内容事件，否则破坏 SSE 事件顺序导致客户端解析混乱
		if streamClosed {
			continue
		}

		// 文本内容 delta
		if deltaContent != "" {
			// 检查当前是否需要新开一个 text block
			if len(blocks) == 0 || blocks[len(blocks)-1].blockType != "text" {
				blocks = append(blocks, &anthropicBlock{
					index:     currentBlockIndex,
					blockType: "text",
					accIndex:  -1,
				})
				currentBlockIndex++
			}
			textBlock := blocks[len(blocks)-1]

			if !textBlock.started {
				textBlock.started = true
				sendSSEEvent(w, flusher, "content_block_start", map[string]interface{}{
					"type":  "content_block_start",
					"index": textBlock.index,
					"content_block": map[string]interface{}{
						"type": "text",
						"text": "",
					},
				})
			}

			sendSSEEvent(w, flusher, "content_block_delta", map[string]interface{}{
				"type":  "content_block_delta",
				"index": textBlock.index,
				"delta": map[string]string{
					"type": "text_delta",
					"text": deltaContent,
				},
			})
		}

		// 检查是否有 tool_calls 累积器更新需要发送 tool_use 块开始
		for i, atc := range accToolCalls {
			if atc == nil || (atc.id == "" && atc.name == "") {
				continue
			}
			// 检查是否已经为此 tool_call 创建了块（按累积器下标精确匹配，防止重复开块）
			existingBlock := false
			for _, b := range blocks {
				if b.blockType == "tool_use" && b.accIndex == i {
					existingBlock = true
					break
				}
			}
			if existingBlock {
				continue
			}

			// 新开 tool_use 块
			useID := atc.id
			useName := atc.name
			if useID == "" {
				useID = fmt.Sprintf("toolu_%d", i)
			}
			blocks = append(blocks, &anthropicBlock{
				index:       currentBlockIndex,
				blockType:   "tool_use",
				toolUseID:   useID,
				toolUseName: useName,
				accIndex:    i,
				started:     true,
			})
			currentBlockIndex++

			sendSSEEvent(w, flusher, "content_block_start", map[string]interface{}{
				"type":  "content_block_start",
				"index": blocks[len(blocks)-1].index,
				"content_block": map[string]interface{}{
					"type":  "tool_use",
					"id":    useID,
					"name":  useName,
					"input": map[string]interface{}{},
				},
			})
		}

		// 发送 tool_use 的 input_json_delta
		// 注意：不能要求 atc.id != "" —— 部分上游流式 delta 不带 id，
		// 此时按累积器索引匹配对应 block（block.accIndex 记录了创建时的累积器下标）
		for i, atc := range accToolCalls {
			if atc == nil || atc.argsBld.Len() == 0 {
				continue
			}
			// 找到对应的 block
			for _, b := range blocks {
				if b.blockType == "tool_use" && b.accIndex == i {
					sendSSEEvent(w, flusher, "content_block_delta", map[string]interface{}{
						"type":  "content_block_delta",
						"index": b.index,
						"delta": map[string]interface{}{
							"type":         "input_json_delta",
							"partial_json": atc.argsBld.String(),
						},
					})
					// 清空已发送的部分
					atc.argsBld.Reset()
					break
				}
			}
		}

		// --- 结束标记 ---
		// 注意：部分上游流式输出 tool_calls 后 finish_reason 返回 "stop" 或缺失，
		// 若仅在 finishReason != nil 时收尾，流正常结束（EOF）但从未收到 finish_reason
		// 会导致 message_stop 永远不发 → 客户端挂起/重试 → 死循环。
		// 收到 finish_reason 时立即收尾；流结束后（循环外）若尚未收尾则强制收尾。
		if finishReason != nil && !streamClosed {
			streamClosed = true
			// 发送所有已开始块的 content_block_stop
			for _, b := range blocks {
				if b.started {
					sendSSEEvent(w, flusher, "content_block_stop", map[string]interface{}{
						"type":  "content_block_stop",
						"index": b.index,
					})
					b.started = false
				}
			}
			stopReason := toAnthropicStopReason(*finishReason)
			// 与非流式 convertOpenAIToAnthropic 保持一致：已发送 tool_use 块时
			// 强制 stop_reason="tool_use"。部分上游（qwen3 中转）返回 tool_calls
			// 却给 finish_reason=stop，客户端会误判"模型没调用工具"而重试 → 死循环
			if stopReason != "tool_use" {
				for _, atc := range accToolCalls {
					if atc != nil && (atc.id != "" || atc.name != "") {
						stopReason = "tool_use"
						break
					}
				}
			}
			sendSSEEvent(w, flusher, "message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason":   stopReason,
					"stop_sequence": nil,
				},
				"usage": map[string]interface{}{
					"output_tokens": outputTokens,
				},
			})
			sendSSEEvent(w, flusher, "message_stop", map[string]interface{}{
				"type": "message_stop",
			})
		}
	}

	// 流结束兜底：上游从未给 finish_reason（EOF/[DONE] 直接结束）时强制收尾，
	// 有 tool_use 块则按 tool_use 收尾，否则按 end_turn
	if msgStarted && !streamClosed {
		streamClosed = true
		for _, b := range blocks {
			if b.started {
				sendSSEEvent(w, flusher, "content_block_stop", map[string]interface{}{
					"type":  "content_block_stop",
					"index": b.index,
				})
				b.started = false
			}
		}
		stopReason := "end_turn"
		for _, atc := range accToolCalls {
			if atc != nil && (atc.id != "" || atc.name != "") {
				stopReason = "tool_use"
				break
			}
		}
		sendSSEEvent(w, flusher, "message_delta", map[string]interface{}{
			"type": "message_delta",
			"delta": map[string]interface{}{
				"stop_reason":   stopReason,
				"stop_sequence": nil,
			},
			"usage": map[string]interface{}{
				"output_tokens": outputTokens,
			},
		})
		sendSSEEvent(w, flusher, "message_stop", map[string]interface{}{
			"type": "message_stop",
		})
	}

	// 零 chunk 兜底：上游返回 200 但一个 delta 都没发（空响应/网关异常），
	// 若什么都不发客户端会挂起并重试 → 死循环。补发一个完整的空消息。
	if !msgStarted {
		sendSSEEvent(w, flusher, "message_start", map[string]interface{}{
			"type": "message_start",
			"message": map[string]interface{}{
				"id":            msgID,
				"type":          "message",
				"role":          "assistant",
				"content":       []interface{}{},
				"model":         areq.Model,
				"stop_reason":   "end_turn",
				"stop_sequence": nil,
				"usage": map[string]interface{}{
					"input_tokens":  inputTokens,
					"output_tokens": 0,
				},
			},
		})
		sendSSEEvent(w, flusher, "message_delta", map[string]interface{}{
			"type": "message_delta",
			"delta": map[string]interface{}{
				"stop_reason":   "end_turn",
				"stop_sequence": nil,
			},
			"usage": map[string]interface{}{
				"output_tokens": 0,
			},
		})
		sendSSEEvent(w, flusher, "message_stop", map[string]interface{}{
			"type": "message_stop",
		})
		fmt.Println("⚠️ Anthropic 流式：上游零 chunk 响应，已补发空消息")
	}

	flusher.Flush()
}

// --- 转换辅助函数 ---

// 发送 SSE 事件
func sendSSEEvent(w http.ResponseWriter, flusher http.Flusher, event string, data interface{}) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(jsonData))
	flusher.Flush()
}

// 生成随机消息 ID
func generateMsgID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}

// 提取 Anthropic 消息中的文本内容
func extractAnthropicContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, block := range v {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}
			if t, _ := blockMap["type"].(string); t == "text" {
				if text, ok := blockMap["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		return fmt.Sprintf("%v", v)
	}
}

// convertAnthropicContentToOpenAI 将 Anthropic 消息内容（含图片/工具块）转为 OpenAI 格式
// Anthropic 图片格式: {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"base64..."}}
// OpenAI 图片格式: {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
// 工具块转换（Agent 工具调用闭环的关键，缺失会导致模型收不到工具结果而反复重试 → 死循环）：
//
//	assistant 的 tool_use 块 → OpenAI assistant.tool_calls
//	user 的 tool_result 块   → OpenAI role:"tool" 消息（tool_call_id + content）
func convertAnthropicContentToOpenAI(content interface{}) interface{} {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		// 检查是否包含图片块
		hasImage := false
		for _, block := range v {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if t, _ := blockMap["type"].(string); t == "image" {
					hasImage = true
					break
				}
			}
		}
		// 没有图片，直接提取文本
		if !hasImage {
			var parts []string
			for _, block := range v {
				if blockMap, ok := block.(map[string]interface{}); ok {
					if t, _ := blockMap["type"].(string); t == "text" {
						if text, ok := blockMap["text"].(string); ok {
							parts = append(parts, text)
						}
					}
				}
			}
			return strings.Join(parts, "\n")
		}
		// 有图片，构建 OpenAI content 数组
		var contentArray []interface{}
		for _, block := range v {
			if blockMap, ok := block.(map[string]interface{}); ok {
				switch blockMap["type"] {
				case "text":
					if text, ok := blockMap["text"].(string); ok && text != "" {
						contentArray = append(contentArray, map[string]interface{}{
							"type": "text",
							"text": text,
						})
					}
				case "image":
					if source, ok := blockMap["source"].(map[string]interface{}); ok {
						mediaType, _ := source["media_type"].(string)
						if mediaType == "" {
							data, _ := source["data"].(string)
							mediaType = detectImageMIME(data)
						}
						data, _ := source["data"].(string)
						if data != "" {
							contentArray = append(contentArray, map[string]interface{}{
								"type": "image_url",
								"image_url": map[string]interface{}{
									"url": "data:" + mediaType + ";base64," + data,
								},
							})
						}
					}
				}
			}
		}
		if len(contentArray) > 0 {
			return contentArray
		}
		return ""
	default:
		return fmt.Sprintf("%v", v)
	}
}

// 转换 Anthropic 请求 → OpenAI 请求
// extractSystemText 从 Anthropic system 字段提取文本（支持 string 和 []{type, text} 两种格式）
func extractSystemText(system interface{}) string {
	if system == nil {
		return ""
	}
	switch v := system.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, block := range v {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if t, _ := blockMap["type"].(string); t == "text" {
					if text, ok := blockMap["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func convertAnthropicToOpenAI(areq *AnthropicReq) ([]byte, error) {
	var messages []map[string]interface{}

	// system 消息放最前面（支持 string 或数组格式）
	systemText := extractSystemText(areq.System)
	if systemText != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": systemText,
		})
	}

	// 转换每条消息（含图片/工具块处理）
	// pendingToolUseIDs 必须在消息循环外声明：Anthropic 协议中 tool_result 块
	// 总在下一条 user 消息里，跨消息才能消费前面 assistant.tool_use 声明的 id
	var pendingToolUseIDs []string
	for _, msg := range areq.Messages {
		content := convertAnthropicContentToOpenAI(msg.Content)

		// 处理工具调用闭环：
		// 1) assistant 消息含 tool_use 块 → 转为 OpenAI tool_calls
		// 2) user 消息含 tool_result 块 → 拆为 role:"tool" 消息（OpenAI 要求 tool 结果独立成条）
		if blocks, ok := msg.Content.([]interface{}); ok {
			var toolUses []map[string]interface{}    // assistant.tool_calls
			var toolResults []map[string]interface{} // role:"tool" 消息
			var textParts []string

			for _, block := range blocks {
				blockMap, ok := block.(map[string]interface{})
				if !ok {
					continue
				}
				switch blockMap["type"] {
				case "tool_use":
					tcID, _ := blockMap["id"].(string)
					name, _ := blockMap["name"].(string)
					// input 是对象，序列化为 JSON 字符串（OpenAI 格式）
					argsBytes, _ := json.Marshal(blockMap["input"])
					toolUses = append(toolUses, map[string]interface{}{
						"id":   tcID,
						"type": "function",
						"function": map[string]interface{}{
							"name":      name,
							"arguments": string(argsBytes),
						},
					})
					// 记录 id 供后续 tool_result 兜底配对（Anthropic 客户端可能省略 id）
					if tcID != "" {
						pendingToolUseIDs = append(pendingToolUseIDs, tcID)
					}
				case "tool_result":
					tcID, _ := blockMap["tool_use_id"].(string)
					// content 可能是 string 或 [{type:"text",text:...}] 数组
					resultText := extractAnthropicContent(blockMap["content"])
					// tool_use_id 缺失时按顺序消费前面 tool_use 声明的 id（兜底配对）
					if tcID == "" && len(pendingToolUseIDs) > 0 {
						tcID = pendingToolUseIDs[0]
						pendingToolUseIDs = pendingToolUseIDs[1:]
					}
					toolResults = append(toolResults, map[string]interface{}{
						"role":         "tool",
						"tool_call_id": tcID,
						"content":      resultText,
					})
				case "text":
					if text, ok := blockMap["text"].(string); ok && text != "" {
						textParts = append(textParts, text)
					}
				}
			}

			// user 消息含 tool_result → 输出 tool 消息（可附带同条消息里的文本）
			if len(toolResults) > 0 {
				for _, tr := range toolResults {
					messages = append(messages, tr)
				}
				if len(textParts) > 0 {
					messages = append(messages, map[string]interface{}{
						"role":    msg.Role,
						"content": strings.Join(textParts, "\n"),
					})
				}
				continue
			}

			// assistant 消息含 tool_use → 转为 tool_calls 格式
			if len(toolUses) > 0 {
				am := map[string]interface{}{
					"role":       "assistant",
					"content":    content,
					"tool_calls": toolUses,
				}
				messages = append(messages, am)
				continue
			}
		}

		messages = append(messages, map[string]interface{}{
			"role":    msg.Role,
			"content": content,
		})
	}

	payload := map[string]interface{}{
		"model":    areq.Model,
		"messages": messages,
		"stream":   areq.Stream,
	}

	if areq.MaxTokens > 0 {
		payload["max_tokens"] = areq.MaxTokens
	}
	if areq.Temperature != nil {
		payload["temperature"] = *areq.Temperature
	}
	if areq.TopP != nil {
		payload["top_p"] = *areq.TopP
	}
	if len(areq.StopSequences) > 0 {
		payload["stop"] = areq.StopSequences
	}
	if len(areq.Tools) > 0 {
		payload["tools"] = areq.Tools
	}
	if areq.ToolChoice != nil {
		payload["tool_choice"] = areq.ToolChoice
	}

	return json.Marshal(payload)
}

// 转换 OpenAI 响应 → Anthropic 响应
func convertOpenAIToAnthropic(raw []byte, model string) ([]byte, error) {
	var upstreamResp map[string]interface{}
	if err := json.Unmarshal(raw, &upstreamResp); err != nil {
		return nil, err
	}

	// 提取 choices[0]
	choices, _ := upstreamResp["choices"].([]interface{})
	finishReason := "end_turn"
	textContent := ""
	var toolUseBlocks []map[string]interface{}
	inputTokens := 0
	outputTokens := 0

	if len(choices) > 0 {
		choice, _ := choices[0].(map[string]interface{})
		if msg, ok := choice["message"].(map[string]interface{}); ok {
			// 提取文本内容
			if c, ok := msg["content"].(string); ok {
				textContent = c
			}

			// 提取 tool_calls → 转为 Anthropic tool_use 内容块
			if tcRaw, ok := msg["tool_calls"]; ok {
				if tcList, ok := tcRaw.([]interface{}); ok {
					tcSeq := 0
					for _, tc := range tcList {
						if tcMap, ok := tc.(map[string]interface{}); ok {
							toolUseID, _ := tcMap["id"].(string)
							// id 缺失时生成兜底 id（Anthropic 客户端要求 tool_use.id 与
							// tool_result.tool_use_id 配对，空 id 会导致死循环）
							if toolUseID == "" {
								tcSeq++
								toolUseID = "toolu_" + generateMsgID() + fmt.Sprintf("_%d", tcSeq)
							}
							if funcRaw, ok := tcMap["function"].(map[string]interface{}); ok {
								name, _ := funcRaw["name"].(string)
								var input interface{} = funcRaw["arguments"]
								// arguments 可能是 JSON 字符串，要反序列化为对象
								if argsStr, ok := funcRaw["arguments"].(string); ok {
									var argsObj interface{}
									if err := json.Unmarshal([]byte(argsStr), &argsObj); err == nil {
										input = argsObj
									}
								}
								toolUseBlocks = append(toolUseBlocks, map[string]interface{}{
									"type":  "tool_use",
									"id":    toolUseID,
									"name":  name,
									"input": input,
								})
							}
						}
					}
				}
			}
		}

		// 提取 finish_reason
		if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
			finishReason = toAnthropicStopReason(fr)
		}
	}

	// 有 tool_use 块但 finish_reason 不是 tool_use 时强制修正：
	// 部分上游（qwen3 中转等）返回 tool_calls 却给 finish_reason=stop，
	// 客户端会误以为模型没调用工具而重试 → 死循环
	if len(toolUseBlocks) > 0 && finishReason != "tool_use" {
		finishReason = "tool_use"
	}

	// 提取 usage
	if usage, ok := upstreamResp["usage"].(map[string]interface{}); ok {
		if pt, ok := usage["prompt_tokens"].(float64); ok {
			inputTokens = int(pt)
		}
		if ct, ok := usage["completion_tokens"].(float64); ok {
			outputTokens = int(ct)
		}
	}

	id := "msg_" + generateMsgID()
	if idStr, ok := upstreamResp["id"].(string); ok && idStr != "" {
		id = idStr
	}

	// 构建 content 数组：文本块 + tool_use 块
	var contentBlocks []interface{}
	if textContent != "" {
		contentBlocks = append(contentBlocks, map[string]interface{}{
			"type": "text",
			"text": textContent,
		})
	}
	for _, tb := range toolUseBlocks {
		contentBlocks = append(contentBlocks, tb)
	}
	if len(contentBlocks) == 0 {
		contentBlocks = []interface{}{}
	}

	ar := map[string]interface{}{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"content":       contentBlocks,
		"model":         model,
		"stop_reason":   finishReason,
		"stop_sequence": nil,
		"usage": map[string]interface{}{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
		},
	}

	return json.Marshal(ar)
}

// OpenAI finish_reason → Anthropic stop_reason 映射
func toAnthropicStopReason(fr string) string {
	switch fr {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "end_turn"
	default:
		return "end_turn"
	}
}

// -------------------- OpenAI API: /v1/models --------------------
func openaiModels(w http.ResponseWriter, r *http.Request) {
	req, _ := http.NewRequest("GET", cfg.OpenAIBase+"/models", nil)
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "upstream error", 500)
		return
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	fmt.Println("UPSTREAM MODELS:", string(raw))

	w.Header().Set("Content-Type", "application/json")
	w.Write(raw)
}

func ollamaVersion(w http.ResponseWriter, r *http.Request) {
	out := map[string]string{
		"version": "0.24.0.0", // VS 只要看到这个字段就会通过
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
	if cfg.Log_Responses {
		fmt.Println("响应内容:", out)
	}
}

func openaiModelsLegacy(w http.ResponseWriter, r *http.Request) {
	req, _ := http.NewRequest("GET", cfg.OpenAIBase+"/models", nil)
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "upstream error", 500)
		return
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	fmt.Println("UPSTREAM LEGACY MODELS:", string(raw))

	w.Header().Set("Content-Type", "application/json")
	w.Write(raw)
}

// -------------------- Ollama API: /api/tags --------------------
func ollamaTags(w http.ResponseWriter, r *http.Request) {
	req, _ := http.NewRequest("GET", cfg.OpenAIBase+"/models", nil)
	req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "upstream error", 500)
		return
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)

	// 先用通用 map 解析上游模型列表，捕获所有可能字段用于提取上下文
	var upstreamModels struct {
		Data []map[string]interface{} `json:"data"`
	}
	json.Unmarshal(raw, &upstreamModels)

	// 从上游获取模型元数据（上下文长度等）
	modelMeta := fetchUpstreamModelMeta()

	// 也尝试从上游模型列表的原始响应中提取字段（有些上游会在 data 数组里直接带元数据）
	for _, modelData := range upstreamModels.Data {
		id, _ := modelData["id"].(string)
		if id == "" {
			continue
		}
		if _, exists := modelMeta[id]; !exists {
			info := upstreamModelMeta{
				ContextLength:   DefaultContextLength,
				MaxOutputTokens: DefaultMaxOutputTokens,
			}
			for _, field := range []string{"context_length", "max_input_tokens", "max_input_length", "context_window", "max_context"} {
				if v := extractNumeric(modelData, field); v > 0 {
					info.ContextLength = v
					break
				}
			}
			for _, field := range []string{"max_output_tokens", "max_completion_tokens", "max_tokens"} {
				if v := extractNumeric(modelData, field); v > 0 {
					info.MaxOutputTokens = v
					break
				}
			}
			modelMeta[id] = info
		}
	}

	const DefaultModelSize = 100 * 1024 * 1024

	var models []map[string]interface{}
	for _, modelData := range upstreamModels.Data {
		modelID, _ := modelData["id"].(string)
		if modelID == "" {
			continue
		}

		// 获取该模型的上文信息
		meta := modelMeta[modelID]
		ctxLen := meta.ContextLength
		maxOut := meta.MaxOutputTokens

		// 显示名称：优先使用 ModelAlias 中的别名，否则用上游 ID，再套上前后缀
		displayName := modelID
		if alias, ok := cfg.ModelAlias[modelID]; ok && alias != "" {
			displayName = alias
		}
		displayName = cfg.OpenAIPrefix + displayName + cfg.OpenAISuffix

		// 获取该模型的能力列表：优先使用 ModelDetailedSettings 中的 Capabilities，否则使用全局 Capabilities
		modelCaps := cfg.Capabilities
		if setting, ok := cfg.ModelDetailedSettings[modelID]; ok && len(setting.Capabilities) > 0 {
			modelCaps = setting.Capabilities
		}

		// 根据模型能力列表动态计算各能力标志
		hasVision := hasCapability(modelCaps, "vision")
		hasTools := hasCapability(modelCaps, "tools")

		models = append(models, map[string]interface{}{
			"name":        displayName, // 显示名（可别名）
			"model":       modelID,     // 实际请求用的模型 ID
			"modelId":     modelID,     // 实际请求用的模型 ID
			"modified_at": time.Now().Format(time.RFC3339),
			"size":        DefaultModelSize,
			"digest":      "sha256:fake",
			"detail":      "Fast, general-purpose model",
			"tooltip":     "This is a tooltip for " + modelID,
			"details": map[string]interface{}{
				"format":             "gguf",
				"family":             modelID,
				"quantization_level": "none",
				"families":           []string{modelID},
			},
			"model_info": map[string]interface{}{
				"general.basename":          displayName,
				"general.architecture":      modelID,
				modelID + ".context_length": ctxLen,
				"num_ctx":                   ctxLen,
				"max_output_tokens":         maxOut,
				"supports_vision":           hasVision,
				"supports_reasoning":        true,
				"supports_tools":            hasTools,
			},
			"capabilities":  modelCaps,
			"contextWindow": ctxLen,
			"options": map[string]interface{}{
				"num_ctx": ctxLen,
			},
			"context_length":                  ctxLen,
			"prompt_tokens":                   ctxLen,
			"completion_tokens":               ctxLen,
			"total_tokens":                    ctxLen,
			"maxInputTokens":                  ctxLen,
			"maxOutputTokens":                 maxOut,
			"capabilities.supports.vision":    hasVision,
			"capabilities.supports.reasoning": true,
			"capabilities.supports.tools":     hasTools,
			"think":                           true,
		})
	}

	out := map[string]interface{}{"models": models}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)

	if cfg.Log_Responses {
		fmt.Println("响应内容:", out)
	}
}

func ollamaShow(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	var req struct {
		Model string `json:"model"`
	}
	json.Unmarshal(body, &req)

	modelID := req.Model
	if modelID == "" {
		modelID = "deepseek-v4-pro"
	}

	// 从上游获取该模型的元数据
	meta := fetchUpstreamModelMeta()
	modelMeta := meta[modelID]
	ctxLen := modelMeta.ContextLength
	maxOut := modelMeta.MaxOutputTokens

	// 获取该模型的能力列表：优先使用 ModelDetailedSettings 中的 Capabilities，否则使用全局 Capabilities
	modelCaps := cfg.Capabilities
	if setting, ok := cfg.ModelDetailedSettings[modelID]; ok && len(setting.Capabilities) > 0 {
		modelCaps = setting.Capabilities
	}
	hasVision := hasCapability(modelCaps, "vision")
	hasTools := hasCapability(modelCaps, "tools")

	// 显示名称：优先使用 ModelAlias 中的别名，否则用上游 ID，再套上前后缀
	displayName := modelID
	if alias, ok := cfg.ModelAlias[modelID]; ok && alias != "" {
		displayName = alias
	}
	displayName = cfg.OpenAIPrefix + displayName + cfg.OpenAISuffix

	// 默认模型大小：1MB（VS2026 只要 >0 就行）
	const DefaultModelSize = 1 * 1024 * 1024

	out := map[string]interface{}{
		"model": map[string]interface{}{
			"name":        displayName,
			"model":       modelID,
			"modelId":     modelID,
			"modified_at": time.Now().Format(time.RFC3339),
			"size":        DefaultModelSize,
			"digest":      "sha256:fake",
			"details": map[string]interface{}{
				"format":             "gguf",
				"family":             modelID,
				"parameter_size":     "1M",
				"quantization_level": "none",
				"families":           []string{modelID},
				"context_length":     ctxLen,
			},
		},
		"model_info": map[string]interface{}{
			"general.basename":          displayName,
			"general.architecture":      modelID,
			modelID + ".context_length": ctxLen,
			"num_ctx":                   ctxLen,
			"max_output_tokens":         maxOut,
			"num_batch":                 512,
			"num_gpu":                   1,
			"general.file_type":         0,
			"llama.context_length":      ctxLen,
			"general.context_length":    ctxLen,
			"n_ctx_train":               ctxLen,
			"context_length":            ctxLen,
		},
		"capabilities":  modelCaps,
		"contextWindow": ctxLen,
		"options": map[string]interface{}{
			"num_ctx": ctxLen,
		},
		"context_length":                  ctxLen,
		"prompt_tokens":                   ctxLen,
		"completion_tokens":               ctxLen,
		"total_tokens":                    ctxLen,
		"maxInputTokens":                  ctxLen,
		"maxOutputTokens":                 maxOut,
		"capabilities.supports.vision":    hasVision,
		"capabilities.supports.reasoning": true,
		"capabilities.supports.tools":     hasTools,
		"think":                           true,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)

	if cfg.Log_Responses {
		fmt.Println("响应内容:", out)
	}
}

// hasImageInBody 检测请求体中是否包含图片
// 支持 Ollama images 数组、Ollama content image 块、Anthropic image 块、OpenAI image_url 四种格式
func hasImageInBody(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return false
	}
	// 检查 messages 数组
	messages, _ := req["messages"].([]interface{})
	for _, msg := range messages {
		msgMap, _ := msg.(map[string]interface{})
		if msgMap == nil {
			continue
		}
		// Ollama images 数组
		if images, _ := msgMap["images"].([]interface{}); len(images) > 0 {
			return true
		}
		// content 数组中的图片块
		if contentArr, _ := msgMap["content"].([]interface{}); len(contentArr) > 0 {
			for _, block := range contentArr {
				blockMap, _ := block.(map[string]interface{})
				if blockMap == nil {
					continue
				}
				switch blockMap["type"] {
				case "image", "image_url":
					return true
				}
			}
		}
	}
	// Anthropic /v1/messages 格式（顶层 content 数组已在 messages 中处理）
	// Anthropic system 字段也可能含图片（较少见，不做额外处理）
	return false
}

// stripModelPrefixSuffix 剥离请求体中的模型名前缀/后缀
// 客户端（VS Code Copilot 等）会用 /api/tags、/v1/models 返回的显示名
// （含 OpenAI_Prefix/OpenAI_Suffix）发起请求，这里还原为上游真实模型 ID。
// 支持 OpenAI /v1/chat/completions、Ollama /api/chat、Anthropic /v1/messages 三种格式。
func stripModelPrefixSuffix(body []byte) []byte {
	if len(body) == 0 || (cfg.OpenAIPrefix == "" && cfg.OpenAISuffix == "") {
		return body
	}

	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body
	}

	model, _ := req["model"].(string)
	if model == "" {
		return body
	}

	cleaned := model
	if cfg.OpenAIPrefix != "" && strings.HasPrefix(cleaned, cfg.OpenAIPrefix) {
		cleaned = strings.TrimPrefix(cleaned, cfg.OpenAIPrefix)
	}
	if cfg.OpenAISuffix != "" && strings.HasSuffix(cleaned, cfg.OpenAISuffix) {
		cleaned = strings.TrimSuffix(cleaned, cfg.OpenAISuffix)
	}
	if cleaned == model {
		return body
	}

	req["model"] = cleaned
	newBody, err := json.Marshal(req)
	if err != nil {
		return body
	}
	fmt.Printf("🔧 模型名还原: %q → %q\n", model, cleaned)
	return newBody
}

// applyRequestPromptReplace 对请求体中的 messages 进行提示词替换
// 根据配置的 RequestPromptReplace 规则，查找并替换指定位置消息中的文本
func applyRequestPromptReplace(body []byte) []byte {
	if len(body) == 0 || len(cfg.RequestPromptReplace) == 0 {
		return body
	}

	// 检查是否有启用的规则（强制替换模式不需要 prompt）
	hasEnabled := false
	for _, rule := range cfg.RequestPromptReplace {
		if rule.Enable && (rule.Prompt != "" || rule.Mode == replaceModeForce) {
			hasEnabled = true
			break
		}
	}
	if !hasEnabled {
		return body
	}

	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body
	}

	modified := false

	// 处理 messages 数组（OpenAI / Ollama chat 格式）
	if rawMessages, ok := req["messages"].([]interface{}); ok {
		for ruleName, rule := range cfg.RequestPromptReplace {
			if !rule.Enable || (rule.Mode != replaceModeForce && rule.Prompt == "") {
				continue
			}

			if rule.Role != "" && rule.Index != nil {
				// ① role + index 都有值：先按 role 过滤，再用 index 取第 N 条
				roleIdx := 0
				for i, rawMsg := range rawMessages {
					msg, ok := rawMsg.(map[string]interface{})
					if !ok {
						continue
					}
					role, _ := msg["role"].(string)
					if role != rule.Role {
						continue
					}
					if roleIdx == *rule.Index {
						// 找到了第 N 条匹配 role 的消息
						if content, ok := msg["content"].(string); ok && (rule.Mode == replaceModeForce || strings.Contains(content, rule.Prompt)) {
							msg["content"] = replaceRuleContent(content, rule)
							modified = true
							fmt.Printf("🔧 提示词替换 [%s]: messages[%d] (role=%q, index=%d)%s\n", ruleName, i, rule.Role, *rule.Index, ruleLogDesc(rule))
						}
						break
					}
					roleIdx++
				}
			} else if rule.Role != "" {
				// ② 只有 role 有值：按 role 过滤所有消息，每条都替换
				for i, rawMsg := range rawMessages {
					if msg, ok := rawMsg.(map[string]interface{}); ok {
						if role, _ := msg["role"].(string); role == rule.Role {
							if content, ok := msg["content"].(string); ok && (rule.Mode == replaceModeForce || strings.Contains(content, rule.Prompt)) {
								msg["content"] = replaceRuleContent(content, rule)
								modified = true
								fmt.Printf("🔧 提示词替换 [%s]: messages[%d] role=%q%s\n", ruleName, i, rule.Role, ruleLogDesc(rule))
							}
						}
					}
				}
			} else if rule.Index != nil && *rule.Index >= 0 && *rule.Index < len(rawMessages) {
				// ③ 只有 index 有值：按 index 取第 N 条消息替换
				idx := *rule.Index
				if msg, ok := rawMessages[idx].(map[string]interface{}); ok {
					if content, ok := msg["content"].(string); ok && (rule.Mode == replaceModeForce || strings.Contains(content, rule.Prompt)) {
						msg["content"] = replaceRuleContent(content, rule)
						modified = true
						fmt.Printf("🔧 提示词替换 [%s]: messages[%d]%s\n", ruleName, idx, ruleLogDesc(rule))
					}
				}
			}
		}
	}

	// 处理 Ollama 单条 prompt 字段
	if prompt, ok := req["prompt"].(string); ok {
		for ruleName, rule := range cfg.RequestPromptReplace {
			if !rule.Enable || (rule.Mode != replaceModeForce && rule.Prompt == "") {
				continue
			}
			if rule.Role == "" && rule.Index != nil && *rule.Index == 0 && (rule.Mode == replaceModeForce || strings.Contains(prompt, rule.Prompt)) {
				req["prompt"] = replaceRuleContent(prompt, rule)
				modified = true
				fmt.Printf("🔧 提示词替换 [%s]: prompt 字段%s\n", ruleName, ruleLogDesc(rule))
			}
		}
	}

	if modified {
		newBody, err := json.Marshal(req)
		if err == nil {
			fmt.Println("✅ 提示词替换完成，请求体已更新")
			return newBody
		}
	}

	return body
}

// replaceRuleContent 根据规则模式决定替换方式：
// force（强制替换）→ 不检查匹配，直接整体覆盖为 Replace；
// whole（匹配整段替换）→ 调用方已确认内容包含 Prompt，整体覆盖为 Replace；
// normal（普通模式）→ 仅将 Prompt 出现处替换为 Replace
func replaceRuleContent(content string, rule PromptReplaceRule) string {
	if rule.Mode == replaceModeForce || rule.Mode == replaceModeWhole {
		return rule.Replace
	}
	return strings.ReplaceAll(content, rule.Prompt, rule.Replace)
}

// ruleLogDesc 生成替换日志的描述片段
func ruleLogDesc(rule PromptReplaceRule) string {
	if rule.Mode == replaceModeForce {
		return " 强制替换整条内容 → \"" + rule.Replace + "\""
	}
	if rule.Mode == replaceModeWhole {
		return " 匹配到后整段替换 → \"" + rule.Replace + "\""
	}
	return " 中 \"" + rule.Prompt + "\" → \"" + rule.Replace + "\""
}

func logAllRequests(w http.ResponseWriter, r *http.Request) {
	count := atomic.AddInt64(&requestCount, 1)
	if count >= cfg.Log_Limit {
		CallClear()
		atomic.StoreInt64(&requestCount, 0)
		fmt.Println("🧹 日志已清理")
	}

	body, _ := io.ReadAll(r.Body)

	hasImage := hasImageInBody(body)
	if hasImage {
		fmt.Println("🖼️ ===== 客户端 请求 (含图片) =====")
	} else {
		fmt.Println("📤 ========= 客户端 请求 ==========")
	}
	fmt.Println("方法:", r.Method)
	fmt.Println("路径:", r.URL.Path)
	fmt.Println("查询:", r.URL.RawQuery)
	if cfg.Log_Headers {
		fmt.Println("Headers:", r.Header)
	}
	if cfg.Log_Body {
		fmt.Println("Body:", string(body))
	}
	fmt.Println("================================")

	// 应用请求提示词替换
	body = applyRequestPromptReplace(body)

	// 剥离模型名前缀/后缀：客户端会用显示名（含 OpenAI_Prefix/OpenAI_Suffix）发起请求，
	// 必须还原为上游真实模型 ID，否则上游报 model_not_found
	body = stripModelPrefixSuffix(body)

	// 视觉代理：主模型不支持图片但配置了 VisionProxyModel 时，
	// 先用代理模型识别图片，再把识别文本合并进请求（图片本身不再转发给主模型）
	if hasImage {
		// 提取请求中的模型 ID
		var reqMeta struct {
			Model string `json:"model"`
		}
		json.Unmarshal(body, &reqMeta)
		if reqMeta.Model != "" {
			if newBody, proxied := applyVisionProxy(body, reqMeta.Model); proxied {
				body = newBody
			}
		}
	}

	// 把 body 放回去，否则后面 handler 读不到
	r.Body = io.NopCloser(bytes.NewBuffer(body))

	// 路由分发
	// 更新模型参数要这样才生效:
	// 1.删掉模型列表,关闭vscode,防止其读取缓存的模型参数
	// 2.api/show 和 api/tags 都要返回新的模型参数,让vscode认为模型列表发生了变化
	// 3.打开vscode,让其获取新的模型参数
	switch {
	case r.URL.Path == "/api/chat":
		ollamaChat(w, r)
	case r.URL.Path == "/api/version":
		ollamaVersion(w, r)
	case r.URL.Path == "/api/tags":
		ollamaTags(w, r)
	case r.URL.Path == "/api/show":
		ollamaShow(w, r)
	case r.URL.Path == "/v1/chat/completions":
		openaiChat(w, r)
	case r.URL.Path == "/v1/messages/count_tokens":
		anthropicCountTokens(w, r)
	case r.URL.Path == "/v1/messages":
		anthropicMessages(w, r)
	case r.URL.Path == "/v1/models":
		openaiModels(w, r)
	case r.URL.Path == "/models": // VSCode 旧版 API
		openaiModelsLegacy(w, r)
	// case r.URL.Path == "/api/show": // VSCode 旧版 API
	// 	ollamaShow(w, r)
	default:
		// 记录未知请求
		fmt.Println("========== 未知请求 ==========")
		fmt.Println("路径:", r.URL.Path)
		fmt.Println("方法:", r.Method)
		fmt.Println("================================")

		// 尝试转发到上游
		if r.URL.Path != "/" {
			req, _ := http.NewRequest(r.Method, cfg.OpenAIBase+r.URL.Path, bytes.NewBuffer(body))
			req.Header.Set("Authorization", "Bearer "+cfg.OpenAIKey)
			req.Header.Set("Content-Type", "application/json")

			client := &http.Client{}
			resp, err := client.Do(req)
			if err == nil {
				defer resp.Body.Close()
				raw, _ := io.ReadAll(resp.Body)
				w.Header().Set("Content-Type", "application/json")
				w.Write(raw)
			} else {
				http.NotFound(w, r)
			}
		} else {
			http.NotFound(w, r)
		}
	}
}

// 设置控制台标题
func setCMDTitle(title string) {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	setConsoleTitleW := kernel32.NewProc("SetConsoleTitleW")
	titlePtr, err := syscall.UTF16PtrFromString(title)
	if err != nil {
		return
	}
	setConsoleTitleW.Call(uintptr(unsafe.Pointer(titlePtr)))
}

// 初始化终端清除函数
func initClear() {
	clear = make(map[string]func()) //初始化它
	clear["linux"] = func() {
		cmd := exec.Command("clear") //Linux 示例，已测试
		cmd.Stdout = os.Stdout
		cmd.Run()
	}
	clear["windows"] = func() {
		cmd := exec.Command("cmd", "/c", "cls") //Windows 示例，已测试
		cmd.Stdout = os.Stdout
		cmd.Run()
	}
}

// 调用终端清除函数
func CallClear() {
	value, ok := clear[runtime.GOOS] //runtime.GOOS -> linux, windows, darwin etc.
	if ok {                          //如果我们为该平台定义了一个明确的函数：
		value() //我们执行它
	} else { //不支持的平台
		panic("您的平台不受支持！我无法清除终端屏幕：(")
	}
}

// 主程序
func main() {
	// 设置窗口标题
	setCMDTitle("🐭 Remote API Convert Ollama by.vancat")

	initClear()
	loadConfig()

	fmt.Println("🐭 Remote API Convert Ollama by.vancat")
	fmt.Println("🔗 上游 OpenAI API: " + cfg.OpenAIBase)
	fmt.Println("🌍 本地 Ollama API: http://" + cfg.IP + ":" + cfg.PORT)
	fmt.Printf("📚 自动清理终端日志: %d 条\n", cfg.Log_Limit)
	fmt.Println("🛡️ 本程序不会保留任何调用记录到本地")

	printConfigHelp()

	printModelAliases()

	fmt.Println("🚀 转换器服务已启动 ~")

	// 注册本地配置管理页面
	http.HandleFunc("/config", handleConfigPage)
	http.HandleFunc("/api/config", handleConfigAPI)
	http.HandleFunc("/api/config/auth", apiConfigAuthStatus)
	http.HandleFunc("/api/config/login", apiConfigLogin)
	http.HandleFunc("/api/config/logout", apiConfigLogout)
	http.HandleFunc("/api/config/test", apiTestConfig)

	webHost := cfg.IP
	if webHost == "" || webHost == "0.0.0.0" {
		webHost = "127.0.0.1"
	}
	pwHint := ""
	if cfg.WebConfigPassword != "" {
		pwHint = "（已启用访问密码 🔒）"
	}
	fmt.Println("⚙️ 配置管理页面: http://" + webHost + ":" + cfg.PORT + "/config " + pwHint)

	http.HandleFunc("/", logAllRequests)
	http.ListenAndServe(cfg.IP+":"+cfg.PORT, nil)
}
