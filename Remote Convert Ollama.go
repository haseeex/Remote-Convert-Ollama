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
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

type ModelTokenSetting struct {
	ContextLength   int64 `json:"ContextLength"`
	MaxOutputTokens int64 `json:"MaxOutputTokens"`
}

type Config struct {
	IP                 string                       `json:"IP"`
	PORT               string                       `json:"PORT"`
	Log_Limit          int64                        `json:"Log_Limit"`
	Log_Responses      bool                         `json:"Log_Responses"`
	Log_Headers        bool                         `json:"Log_Headers"`
	Log_Body           bool                         `json:"Log_Body"`
	OpenAIPrefix       string                       `json:"OpenAI_Prefix"`
	OpenAISuffix       string                       `json:"OpenAI_Suffix"`
	StreamMode         string                       `json:"StreamMode"`
	Capabilities       []string                     `json:"Capabilities"`
	OpenAIBase         string                       `json:"OPENAI_BASE"`
	OpenAIKey          string                       `json:"OPENAI_KEY"`
	ModelAlias         map[string]string            `json:"ModelAlias"`
	ModelTokenSettings map[string]ModelTokenSetting `json:"ModelTokenSettings"`
}

var requestCount int64
var clear map[string]func() //创建一个用于存储清除函数的映射

var cfg Config

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
type OllamaToolCall struct {
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
func convertMessagesToOpenAI(messages []interface{}) []interface{} {
	var result []interface{}
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			result = append(result, msg)
			continue
		}

		role, _ := msgMap["role"].(string)

		// 处理 tool 角色消息：添加 tool_call_id 如果不存在
		if role == "tool" {
			newMsg := make(map[string]interface{})
			for k, v := range msgMap {
				newMsg[k] = v
			}
			if _, hasID := newMsg["tool_call_id"]; !hasID {
				newMsg["tool_call_id"] = ""
			}
			result = append(result, newMsg)
			continue
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
					openaiTCs = append(openaiTCs, openaiTC)
				} else {
					// Ollama 格式（无 id）：从 function 字段构建
					funcRaw, ok := tcMap["function"]
					if !ok {
						continue
					}
					funcMap, ok := funcRaw.(map[string]interface{})
					if !ok {
						continue
					}

					openaiTC := map[string]interface{}{
						"id":   "", // 将由上游 API 分配，但留空
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
		IP:                 "0.0.0.0",
		PORT:               "11434",
		Log_Limit:          100,
		Log_Responses:      true,
		Log_Headers:        true,
		Log_Body:           true,
		OpenAIPrefix:       "[VC反代] ",
		OpenAISuffix:       "",
		StreamMode:         streamModePreserve,
		Capabilities:       []string{"tools", "vision"}, // vs2026 需要这个字段才能启用工具功能
		OpenAIBase:         "https://api.openai.com/v1",
		OpenAIKey:          "",
		ModelAlias:         map[string]string{},            // 模型别名：key=上游模型ID, value=显示名称
		ModelTokenSettings: map[string]ModelTokenSetting{}, // 模型 token 设置：key=上游模型ID, value={ContextLength, MaxOutputTokens}
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
	fmt.Println(" ▼ ModelTokenSettings : 模型 Token 手动设置,覆盖上游自动获取的值")
	fmt.Println("                     格式: {上游模型ID: {ContextLength: 上下文长度, MaxOutputTokens: 最大输出}, ...}")
	fmt.Println("                     示例: {\"gpt-4o\": {\"ContextLength\": 128000, \"MaxOutputTokens\": 16384}}")
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
		fmt.Printf("       🛠️  能力集合:   %v\n", cfg.Capabilities)
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
	if _, ok := rawMap["ModelTokenSettings"]; !ok {
		stored.ModelTokenSettings = defaultCfg.ModelTokenSettings
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

	cfg = stored
	cfg.OpenAIKey = plainKey

	if persistedKey != "" {
		stored.OpenAIKey = persistedKey
		if err := saveConfig(stored); err != nil {
			fmt.Println("🔒 OPENAI_KEY 自动回写失败:", err)
			pauseAndExit()
		}
		fmt.Println("🔒 OPENAI_KEY 已按本机信息加密并回写到 config.json")
	}
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

// upstreamModelMeta 上游模型元数据
type upstreamModelMeta struct {
	ContextLength   int64
	MaxOutputTokens int64
}

// fetchUpstreamModelMeta 调用上游 /v1/models 获取模型元数据并构建映射，
// 然后合并 ModelTokenSettings 中手动指定的值（手动设置优先）
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

	// 合并 ModelTokenSettings 手动配置（手动设置优先）
	return applyManualModelSettings(result)
}

// applyManualModelSettings 将 ModelTokenSettings 中的手动配置合并到 result 中
func applyManualModelSettings(result map[string]upstreamModelMeta) map[string]upstreamModelMeta {
	for modelID, setting := range cfg.ModelTokenSettings {
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

	out := map[string]interface{}{
		"model":             model,
		"created_at":        time.Now().Format("2006-01-02T15:04:05"),
		"message":           makeOllamaMessage("assistant", content, toolCalls, reasoningContent),
		"done":              true,
		"done_reason":       mapFinishReason(finishReason),
		"total_duration":    1,
		"load_duration":     1,
		"prompt_eval_count": 1,
		"eval_count":        1,
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
	isToolCallFinish := false
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
		finishReason := choice.FinishReason

		// 保存上游返回的 finish_reason（用于最终 done_reason）
		if finishReason != nil && *finishReason != "" {
			upstreamFinishReason = *finishReason
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
					// 实时推送 thinking 增量（sendOllamaChunk 内部自动计算增量）
					sendOllamaChunk("", false, 0, nil, reasoningContent.String())
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
			sendOllamaChunk(content, false, 0, nil, reasoningContent.String())
		}

		// 检查是否 tool_calls 结束
		if finishReason != nil && *finishReason == "tool_calls" {
			isToolCallFinish = true
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

	// 构建最终消息
	if hasToolCalls && isToolCallFinish {
		// 将累积的 tool_calls 转为 Ollama 格式
		var ollamaTCs []OllamaToolCall
		for _, atc := range accToolCalls {
			if atc == nil {
				continue
			}
			var otc OllamaToolCall
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
						// thinking 结束后，从 delta 中移除 reasoning_text（如果有）
						delete(delta, "reasoning_text")
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

	// 内容块跟踪：0=text, 1=tool_use...
	type anthropicBlock struct {
		index       int
		blockType   string // "text" or "tool_use"
		started     bool
		toolUseID   string
		toolUseName string
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

		// 文本内容 delta
		if deltaContent != "" {
			// 检查当前是否需要新开一个 text block
			if len(blocks) == 0 || blocks[len(blocks)-1].blockType != "text" {
				blocks = append(blocks, &anthropicBlock{
					index:     currentBlockIndex,
					blockType: "text",
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
			// 检查是否已经为此 tool_call 创建了块
			existingBlock := false
			for _, b := range blocks {
				if b.blockType == "tool_use" && b.toolUseID == atc.id && b.toolUseName == atc.name && atc.id != "" {
					existingBlock = true
					break
				}
			}
			if existingBlock {
				continue
			}

			// 检查之前是否已经为这个索引创建了块（通过 id 空匹配）
			if atc.id == "" && atc.name != "" {
				// 如果有任何 tool_use 块已存在，跳过
				for _, b := range blocks {
					if b.blockType == "tool_use" && b.index == i+100 {
						existingBlock = true
						break
					}
				}
				if existingBlock {
					continue
				}
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
		for _, atc := range accToolCalls {
			if atc == nil || atc.argsBld.Len() == 0 {
				continue
			}
			// 找到对应的 block
			for _, b := range blocks {
				if b.blockType == "tool_use" && b.toolUseID == atc.id && atc.id != "" {
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
		if finishReason != nil {
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
			sendSSEEvent(w, flusher, "message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason":   toAnthropicStopReason(*finishReason),
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

// convertAnthropicContentToOpenAI 将 Anthropic 消息内容（含图片）转为 OpenAI 格式
// Anthropic 图片格式: {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"base64..."}}
// OpenAI 图片格式: {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
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

	// 转换每条消息（含图片处理）
	for _, msg := range areq.Messages {
		content := convertAnthropicContentToOpenAI(msg.Content)
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
					for _, tc := range tcList {
						if tcMap, ok := tc.(map[string]interface{}); ok {
							toolUseID, _ := tcMap["id"].(string)
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
				"supports_vision":           true,
				"supports_reasoning":        true,
				"supports_tools":            true,
			},
			"capabilities":  cfg.Capabilities,
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
			"capabilities.supports.vision":    true,
			"capabilities.supports.reasoning": true,
			"capabilities.supports.tools":     true,
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
		"capabilities":  cfg.Capabilities,
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
		"capabilities.supports.vision":    true,
		"capabilities.supports.reasoning": true,
		"capabilities.supports.tools":     true,
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

	http.HandleFunc("/", logAllRequests)
	http.ListenAndServe(cfg.IP+":"+cfg.PORT, nil)
}
