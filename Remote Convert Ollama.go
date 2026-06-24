/*
接口提供的字段兼容性优先适配 Visual Studio Code Copilot Chat
https://github.com/microsoft/vscode-copilot-chat/blob/main/src/extension/byok/vscode-node/ollamaProvider.ts#L137C2-L137C124
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
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

type Config struct {
	IP           string            `json:"IP"`
	PORT         string            `json:"PORT"`
	Log_Limit    int64             `json:"Log_Limit"`
	OpenAIPrefix string            `json:"OpenAI_Prefix"`
	OpenAISuffix string            `json:"OpenAI_Suffix"`
	EnableStream bool              `json:"EnableStream"`
	Capabilities []string          `json:"Capabilities"`
	OpenAIBase   string            `json:"OPENAI_BASE"`
	OpenAIKey    string            `json:"OPENAI_KEY"`
	ModelAlias   map[string]string `json:"ModelAlias"`
}

var requestCount int64
var clear map[string]func() //创建一个用于存储清除函数的映射

var cfg Config

const encryptedKeyPrefix = "已加密|"

// 这个 UUID 是用来增强加密安全性的，确保同一台机器上的加密结果不同于其他机器。它不会泄露任何敏感信息。
// 推荐生成网站 https://www.uuidgenerator.net/ 生成一个随机的 UUID 来替换这个值。
const secretUUID = "vancat-10a8bca6-fe6f-4bcd-8c9a-9a27d6ec1b16"

type Choice struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Text  string `json:"text"`
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
}

type OpenAIResp struct {
	Choices []Choice `json:"choices"`
}

// -------------------- Anthropic Messages API 类型 --------------------

type AnthropicReq struct {
	Model         string                 `json:"model"`
	Messages      []AnthropicMessage     `json:"messages"`
	System        string                 `json:"system,omitempty"`
	MaxTokens     int                    `json:"max_tokens"`
	Stream        bool                   `json:"stream,omitempty"`
	Temperature   *float64               `json:"temperature,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
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

func extractContent(resp *OpenAIResp) string {
	if len(resp.Choices) == 0 {
		return ""
	}
	ch := resp.Choices[0]

	if ch.Message.Content != "" {
		return ch.Message.Content
	}
	if ch.Text != "" {
		return ch.Text
	}
	if ch.Delta.Content != "" {
		return ch.Delta.Content
	}
	return ""
}

func getDefaultConfig() Config {
	return Config{
		IP:           "0.0.0.0",
		PORT:         "11434",
		Log_Limit:    100,
		OpenAIPrefix: "[VC反代] ",
		OpenAISuffix: "",
		EnableStream: true,
		Capabilities: []string{"tools", "vision"}, // vs2026 需要这个字段才能启用工具功能
		OpenAIBase:   "https://api.openai.com/v1",
		OpenAIKey:    "",
		ModelAlias:   map[string]string{}, // 模型别名：key=上游模型ID, value=显示名称
	}
}

func printConfigHelp() {
	fmt.Println("")
	fmt.Println("══════════════════════════════════════════════ 🪄 配置项说明 ═════════════════════════════════════════════════")
	fmt.Println(" ▼ IP              : 监听地址 (默认 0.0.0.0，本机测试用 127.0.0.1)")
	fmt.Println(" ▼ PORT            : 监听端口 (默认 11434，即 Ollama 默认端口)")
	fmt.Println(" ▼ Log_Limit       : 终端自动清理的日志行数阈值")
	fmt.Println(" ▼ OpenAI_Prefix   : 返回给客户端的模型名称前缀,仅影响模型名字显示")
	fmt.Println(" ▼ OpenAI_Suffix   : 返回给客户端的模型名称后缀,仅影响模型名字显示")
	fmt.Println(" ▼ EnableStream    : 是否启用流式传输 (true/false)")
	fmt.Println(" ▼ Capabilities    : 向客户端声明支持的能力列表 (tools, vision 等)")
	fmt.Println(" ▼ OPENAI_BASE     : 上游 OpenAI 兼容 API 地址 (必填)")
	fmt.Println(" ▼ OPENAI_KEY      : 上游 API 密钥 (必填，每次启动时自动加密存储,换设备需重新输入)")
	fmt.Println(" ▼ ModelAlias      : 模型别名映射,仅影响模型名字显示 {上游模型ID: 显示名称, 上游模型ID: 显示名称, ...}")
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

	fmt.Println("📋 上游拥有的模型:")
	for _, m := range upstream.Data {
		if alias, ok := cfg.ModelAlias[m.ID]; ok && alias != "" {
			fmt.Printf("   🧩 %s → %s%s%s\n", m.ID, cfg.OpenAIPrefix, alias, cfg.OpenAISuffix)
		} else {
			fmt.Printf("   💠 %s → %s%s%s\n", m.ID, cfg.OpenAIPrefix, m.ID, cfg.OpenAISuffix)
		}
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
	if _, ok := rawMap["OpenAI_Prefix"]; !ok {
		stored.OpenAIPrefix = defaultCfg.OpenAIPrefix
		needSave = true
	}
	if _, ok := rawMap["OpenAI_Suffix"]; !ok {
		stored.OpenAISuffix = defaultCfg.OpenAISuffix
		needSave = true
	}
	if _, ok := rawMap["EnableStream"]; !ok {
		stored.EnableStream = defaultCfg.EnableStream
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

	return encryptedKeyPrefix + fingerprint + "|" + base64.StdEncoding.EncodeToString(payload) + "|" + secretUUID, nil
}

func decryptOpenAIKey(value string) (string, error) {
	parts := strings.SplitN(value, "|", 4)
	if len(parts) != 4 {
		return "", errors.New("🔒 已加密格式不正确")
	}

	fingerprint, err := getMachineFingerprint()
	if err != nil {
		return "", err
	}
	if parts[1] != fingerprint {
		return "", errors.New("🔒 机器码不匹配，需要重新输入 OPENAI_KEY")
	}
	if parts[3] != secretUUID {
		return "", errors.New("🔒 双重因素UUID 不匹配，需要重新输入 OPENAI_KEY")
	}

	payload, err := base64.StdEncoding.DecodeString(parts[2])
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

// -------------------- Ollama API: /api/chat --------------------
func ollamaChat(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	fmt.Println("OLLAMA /api/chat REQ:", string(body))

	var req map[string]interface{}
	json.Unmarshal(body, &req)

	model := "deepseek-chat"
	if m, ok := req["model"].(string); ok {
		model = m
	}

	messages := req["messages"]
	if messages == nil {
		messages = []map[string]string{
			{"role": "user", "content": req["prompt"].(string)},
		}
	}

	payload := map[string]interface{}{
		"model":    model,
		"messages": messages,
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

	var oai OpenAIResp
	json.Unmarshal(raw, &oai)

	content := extractContent(&oai)

	out := map[string]interface{}{
		"model":             model,
		"created_at":        time.Now().Format("2006-01-02T15:04:05"),
		"message":           map[string]string{"role": "assistant", "content": content},
		"done":              true,
		"total_duration":    1,
		"load_duration":     1,
		"prompt_eval_count": 1,
		"eval_count":        1,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
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

	// 非流式请求 - 只打印元数据
	var meta struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &meta); err == nil {
		msgCount := len(meta.Messages)
		totalLen := 0
		for _, m := range meta.Messages {
			totalLen += len(m.Content)
		}
		fmt.Printf("VS /v1/chat/completions REQ: model=%s, messages=%d, total_chars=%d, stream=false\n",
			meta.Model, msgCount, totalLen)
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

	w.Header().Set("Content-Type", "application/json")
	w.Write(raw)
}

// 流式响应处理
func openaiChatStream(w http.ResponseWriter, r *http.Request, body []byte) {
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

	// 设置流式响应头
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				// 发送结束标记
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
			}
			break
		}

		fmt.Fprint(w, line)
		flusher.Flush()
	}
}

// -------------------- Anthropic Messages API: /v1/messages --------------------
func anthropicMessages(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	var areq AnthropicReq
	if err := json.Unmarshal(body, &areq); err != nil {
		http.Error(w, `{"error":{"type":"invalid_request_error","message":"Invalid JSON"}}`, 400)
		return
	}

	if areq.Stream {
		anthropicMessagesStream(w, r, &areq)
		return
	}

	// --- 非流式请求 ---
	fmt.Printf("VS /v1/messages REQ: model=%s, messages=%d, max_tokens=%d, stream=false\n",
		areq.Model, len(areq.Messages), areq.MaxTokens)

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

// --- 流式处理 ---
func anthropicMessagesStream(w http.ResponseWriter, r *http.Request, areq *AnthropicReq) {
	fmt.Printf("VS /v1/messages REQ: model=%s, messages=%d, max_tokens=%d, stream=true\n",
		areq.Model, len(areq.Messages), areq.MaxTokens)

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
	blockStarted := false
	allContent := ""

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

		var chunk OpenAIChunk
		if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
			continue
		}

		// 获取 usage（如果有）
		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens > 0 {
				inputTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				outputTokens = chunk.Usage.CompletionTokens
			}
		}

		// 提取 content
		content := ""
		var finishReason *string
		if len(chunk.Choices) > 0 {
			content = chunk.Choices[0].Delta.Content
			finishReason = chunk.Choices[0].FinishReason
		}

		// 第一次有内容 → message_start
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
					"usage":         AnthropicUsage{InputTokens: inputTokens, OutputTokens: 0},
				},
			})
		}

		// 第一次有实际内容 → content_block_start
		if content != "" && !blockStarted {
			blockStarted = true
			sendSSEEvent(w, flusher, "content_block_start", map[string]interface{}{
				"type":  "content_block_start",
				"index": 0,
				"content_block": map[string]interface{}{
					"type": "text",
					"text": "",
				},
			})
		}

		// 发送内容增量
		if content != "" {
			allContent += content
			sendSSEEvent(w, flusher, "content_block_delta", map[string]interface{}{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]string{
					"type": "text_delta",
					"text": content,
				},
			})
		}

		// 结束标记
		if finishReason != nil {
			if blockStarted {
				sendSSEEvent(w, flusher, "content_block_stop", map[string]interface{}{
					"type":  "content_block_stop",
					"index": 0,
				})
			}
			sendSSEEvent(w, flusher, "message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason":   toAnthropicStopReason(*finishReason),
					"stop_sequence": nil,
				},
				"usage": AnthropicUsage{OutputTokens: outputTokens},
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

// 转换 Anthropic 请求 → OpenAI 请求
func convertAnthropicToOpenAI(areq *AnthropicReq) ([]byte, error) {
	var messages []map[string]interface{}

	// system 消息放最前面
	if areq.System != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": areq.System,
		})
	}

	// 转换每条消息
	for _, msg := range areq.Messages {
		content := extractAnthropicContent(msg.Content)
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

	return json.Marshal(payload)
}

// 转换 OpenAI 响应 → Anthropic 响应
func convertOpenAIToAnthropic(raw []byte, model string) ([]byte, error) {
	var oai OpenAIUsageResp
	if err := json.Unmarshal(raw, &oai); err != nil {
		return nil, err
	}

	content := ""
	finishReason := "end_turn"
	if len(oai.Choices) > 0 {
		content = oai.Choices[0].Message.Content
		fr := oai.Choices[0].FinishReason
		if fr != "" {
			finishReason = toAnthropicStopReason(fr)
		}
	}

	id := "msg_" + generateMsgID()
	if oai.ID != "" {
		id = oai.ID
	}

	ar := AnthropicResp{
		ID:           id,
		Type:         "message",
		Role:         "assistant",
		Model:        model,
		StopReason:   finishReason,
		StopSequence: nil,
		Usage: AnthropicUsage{
			InputTokens:  oai.Usage.PromptTokens,
			OutputTokens: oai.Usage.CompletionTokens,
		},
	}

	if content != "" {
		ar.Content = []AnthropicRespBlock{{Type: "text", Text: content}}
	} else {
		ar.Content = []AnthropicRespBlock{}
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
	fmt.Println("响应内容:", out)
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

	var oai struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	json.Unmarshal(raw, &oai)

	const DefaultModelSize = 100 * 1024 * 1024

	var models []map[string]interface{}
	for _, m := range oai.Data {
		// 显示名称：优先使用 ModelAlias 中的别名，否则用上游 ID，再套上前后缀
		displayName := m.ID
		if alias, ok := cfg.ModelAlias[m.ID]; ok && alias != "" {
			displayName = alias
		}
		displayName = cfg.OpenAIPrefix + displayName + cfg.OpenAISuffix
		models = append(models, map[string]interface{}{
			"name":        displayName, // 显示名（可别名）
			"model":       m.ID,        // 实际请求用的模型 ID
			"modelId":     m.ID,        // 实际请求用的模型 ID
			"modified_at": time.Now().Format(time.RFC3339),
			"size":        DefaultModelSize,
			"digest":      "sha256:fake",
			"detail":      "Fast, general-purpose model",
			"tooltip":     "This is a tooltip for " + m.ID,
			"details": map[string]interface{}{
				"format":             "gguf",
				"family":             m.ID,
				"quantization_level": "none",
				"families":           []string{m.ID}, // ← 加这个
			},
			"model_info": map[string]interface{}{ // ← 加这个
				"general.basename":       displayName, // ← VC Code 用这个字段显示别名
				"general.architecture":   m.ID,
				m.ID + ".context_length": 1000000, // ← 动态生成
				"num_ctx":                1000000, // 1M 上下文
				"max_output_tokens":      1000000,
				"supports_vision":        true,
				"supports_reasoning":     true, // ← 加这个
				"supports_tools":         true, // ← 加这个
			},
			"capabilities":  cfg.Capabilities, // vs2026 需要这个字段才能启用工具功能
			"contextWindow": 1000000,          // 65536 上下文
			"options": map[string]interface{}{
				"num_ctx": 1000000,
			},
			"context_length":                  1000000, // 1M 上下文
			"prompt_tokens":                   1000000, // 1M 上下文
			"completion_tokens":               1000000, // 1M 上下文
			"total_tokens":                    1000000, // 1M 上下文
			"maxInputTokens":                  1000000, // 1M 上下文
			"maxOutputTokens":                 384000,  // 1M 上下文
			"capabilities.supports.vision":    true,
			"capabilities.supports.reasoning": true, // ← 加这个
			"capabilities.supports.tools":     true, // ← 加这个
			"think":                           true, // ← 加这个，启用 VS2026 的思考功能
		})
	}

	out := map[string]interface{}{"models": models}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)

	fmt.Println("响应内容:", out)
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
			"name":        displayName, // 显示名（可别名）
			"model":       modelID,     // 实际请求用的模型 ID
			"modelId":     modelID,     // 实际请求用的模型 ID
			"modified_at": time.Now().Format(time.RFC3339),
			"size":        DefaultModelSize,
			"digest":      "sha256:fake",
			"details": map[string]interface{}{
				"format":             "gguf",
				"family":             modelID,
				"parameter_size":     "1M",
				"quantization_level": "none",
				"families":           []string{modelID}, // ← 加这个
				"context_length":     1000000,           // 1M 上下文
			},
		},
		"model_info": map[string]interface{}{
			"general.basename":          displayName, // ← VC Code 用这个字段显示别名
			"general.architecture":      modelID,     // ← 用 name
			modelID + ".context_length": 1000000,     // ← 加这个动态字段！
			"num_ctx":                   1000000,     // DeepSeek V4P 真实上下文：1M tokens
			"max_output_tokens":         65535,       // DeepSeek V4P 最大输出：65535 tokens
			"num_batch":                 512,
			"num_gpu":                   1,
			//"general.architecture":   "deepseek",
			"general.file_type":      0,
			"llama.context_length":   1000000, // 1M 上下文
			"general.context_length": 1000000, // 1M 上下文
			"n_ctx_train":            1000000, // 1M 上下文
			"context_length":         1000000, // 1M 上下文
		},
		"capabilities":  cfg.Capabilities, // vs2026 需要这个字段才能启用工具功能
		"contextWindow": 1000000,          // 65536 上下文
		"options": map[string]interface{}{
			"num_ctx": 1000000,
		},
		"context_length":                  1000000, // 1M 上下文
		"prompt_tokens":                   1000000, // 1M 上下文
		"completion_tokens":               1000000, // 1M 上下文
		"total_tokens":                    1000000, // 1M 上下文
		"maxInputTokens":                  1000000, // 1M 上下文
		"maxOutputTokens":                 384000,  // 1M 上下文
		"capabilities.supports.vision":    true,
		"capabilities.supports.reasoning": true, // ← 加这个
		"capabilities.supports.tools":     true, // ← 加这个
		"think":                           true, // ← 加这个，启用 VS2026 的思考功能
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)

	fmt.Println("响应内容:", out)
}

func logAllRequests(w http.ResponseWriter, r *http.Request) {
	count := atomic.AddInt64(&requestCount, 1)
	if count >= cfg.Log_Limit {
		CallClear()
		atomic.StoreInt64(&requestCount, 0)
		fmt.Println("🧹 日志已清理")
	}

	body, _ := io.ReadAll(r.Body)

	fmt.Println("📤 ========= 客户端 请求 ==========")
	fmt.Println("方法:", r.Method)
	fmt.Println("路径:", r.URL.Path)
	fmt.Println("查询:", r.URL.RawQuery)
	fmt.Println("Body:", string(body))
	fmt.Println("Headers:", r.Header)
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
