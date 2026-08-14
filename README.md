<p align="center">
  <img src="https://img.shields.io/badge/Go-1.21+-00ADD8?style=for-the-badge&logo=go&logoColor=white" alt="Go Version"/>
  <img src="https://img.shields.io/badge/VS%20Code-Compatible-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" alt="VS Code"/>
  <img src="https://img.shields.io/badge/VS2026-Compatible-5C2D91?style=for-the-badge&logo=visualstudio&logoColor=white" alt="VS2026"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge&logo=apache&logoColor=white" alt="License"/>
</p>

<h1 align="center">🐭 Remote API Convert Ollama</h1>

<p align="center">
  <b>将任意 OpenAI 兼容 API 转换为 Ollama API / Anthropic API 的本地反代网关</b><br>
  <sub>让 VS Code Copilot 和 VS2026 能够使用第三方 OpenAI 兼容接口</sub>
</p>

<p align="center">
  <b>因为微软不支持第三方 API，所以我造了一个轮子 🛞</b>
</p>

<p align="center">
  <img src="images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-07-28%20083154.png" alt="Remote API Convert Ollama 预览" width="400"/>
</p>

---

## 📖 概述

**Remote API Convert Ollama** 是一个用 Go 编写的轻量级本地反向代理服务器。它监听在本地（或局域网）的 Ollama 兼容端口上，将 **Ollama API** 和 **Anthropic Messages API** 的请求实时转换为 **OpenAI 兼容 API** 请求，并转发到上游服务。

> 💡 **简单来说**：你只需在 VS Code / VS2026 中配置 Ollama 作为 API 提供商，然后指向本程序，就能使用任何 **OpenAI 兼容的 API 服务**（如 DeepSeek、GPT、Claude 等）。

---

## 🎯 为什么需要这个工具？

| 问题                                                   | 解决方案                                                     |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| ❌ VS Code Copilot Chat 只支持 Ollama API 调用本地模型 | ✅ 本程序模拟 Ollama API，实际调用远程 OpenAI 兼容 API       |
| ❌ VS2026 仅内置支持 OpenAI + Azure + Anthropic 官方   | ✅ 同时提供 Ollama API + Anthropic Messages API 两种接入方式 |
| ❌ 官方限制多、地区不可用、价格高昂                    | ✅ 自由选择任意第三方 OpenAI 兼容服务商                      |
| ❌ API Key 明文存储有泄露风险                          | ✅ AES-GCM 加密存储，绑定机器指纹 + UUID 双重校验            |

---

## ✨ 核心功能

### 🔄 多协议转换

| 客户端请求                         | 转换目标                               | 说明                                        |
| ---------------------------------- | -------------------------------------- | ------------------------------------------- |
| `GET /api/version`               | → 返回版本信息                        | VS Code 探测 Ollama 服务                    |
| `GET /api/tags`                  | →`GET /v1/models` (上游)            | 获取模型列表，支持别名、前后缀、上下文信息  |
| `POST /api/show`                 | → 返回增强模型信息                    | 包含上下文窗口、能力声明、Token 限制等      |
| `POST /api/chat`                 | →`POST /v1/chat/completions` (上游) | Ollama 聊天补全 → OpenAI 格式              |
| `POST /v1/chat/completions`      | →`POST /v1/chat/completions` (上游) | 标准 OpenAI 流式/非流式透传                 |
| `POST /v1/messages`              | →`POST /v1/chat/completions` (上游) | **Anthropic 格式 → OpenAI 格式转换** |
| `POST /v1/messages/count_tokens` | → Token 估算                          | Anthropic token 计数接口                    |
| `GET /v1/models`                 | →`GET /v1/models` (上游)            | 获取上游模型列表                            |
| `GET /models`                    | →`GET /v1/models` (上游)            | VS Code 旧版 API 兼容                       |

### 🖥️ VS Code & VS2026 完美兼容

- ✅ **Capabilities 声明** — 向客户端声明支持 `tools`、`vision` 等能力
- ✅ **模型别名系统** — 通过 `ModelAlias` 配置将上游模型 ID 映射为友好名称
- ✅ **显示名前缀/后缀** — 在客户端看到类似 `[VC反代] 高级智商` 的模型名称
- ✅ **思考功能** — 返回 `think: true` 启用推理能力
- ✅ **Reasoning 内容追踪** — 自动追踪 DeepSeek 思考模式，后续请求自动注入 `reasoning_content`
- ✅ **VS Code 兼容映射** — `reasoning_content` → `reasoning_text` 自动映射
- ✅ **超大上下文** — 声明 1M tokens 上下文窗口，支持每模型独立设置
- ✅ **流式传输 (SSE)** — 支持 `stream: true`，实时输出

### 🛠️ 工具调用支持

- **Ollama 格式** → 自动转换为 OpenAI tool_calls 格式转发
- **OpenAI 流式 tool_calls** → 累积合并后输出完整 tool_calls
- **Anthropic tool_use** → 流式 `input_json_delta` 事件支持
- **Anthropic 非流式** → tool_use content block 转换

### 🔒 安全保障

- **AES-GCM 加密存储**：API Key 首次输入后自动加密，配置文件不留明文
- **机器指纹绑定**：主机名 + 系统盘序列号 + OS + 架构 → SHA256 指纹
- **双重密钥校验**：机器指纹 + 自定义 UUID 双重解密校验
- **跨设备失效**：加密后的配置文件换机自动失效，需重新输入 Key
- **日志无残留**：程序不会将任何调用记录写入本地文件

### ⚙️ 智能配置

- **自动创建**：首次运行自动生成 `config.json`
- **自动补全**：版本更新后自动补充新增配置项
- **自动加密**：首次输入明文 Key 自动加密回写
- **自动获取模型列表**：启动时显示上游所有可用模型及其别名映射、上下文长度、最大输出
- **流式策略**：支持 `preserve` / `force_stream` / `force_close` 三种模式灵活切换
- **日志分级**：可分别控制请求头(`Log_Headers`)、请求体(`Log_Body`)、响应内容(`Log_Responses`)的日志打印
- **模型详细设置**：`ModelDetailedSettings` 支持为每个模型单独指定上下文长度、最大输出 Token 和能力列表
- **请求提示词替换**：`RequestPromptReplace` 支持自动替换请求消息中的指定文本，实现 Copilot 自有提示词篡改等高级玩法

---

## 📦 安装

### 方法一：直接下载

从 [Releases](https://github.com/haseeex/Remote-Convert-Ollama/releases) 页面下载预编译的 `Remote Convert Ollama.exe`。

### 方法二：自行编译

```bash
# 克隆仓库
git clone https://github.com/haseeex/Remote-Convert-Ollama.git
cd Remote-Convert-Ollama

# 安装 garble（用于混淆编译，可选）
go install mvdan.cc/garble@latest

# 混淆编译（推荐）
garble build -o "Remote Convert Ollama.exe" "Remote Convert Ollama.go"

# 或者直接编译
go build -o "Remote Convert Ollama.exe" "Remote Convert Ollama.go"
```

---

## 🚀 快速开始

### 1️⃣ 配置 `config.json`

首次运行会自动生成 `config.json`，编辑它：

```json
{
    "IP": "127.0.0.1",
    "PORT": "11434",
    "Log_Limit": 100,
    "Log_Responses": true,
    "Log_Headers": true,
    "Log_Body": true,
    "OpenAI_Prefix": "[VC反代] ",
    "OpenAI_Suffix": "",
    "StreamMode": "preserve",
    "Capabilities": [
        "tools",
        "vision"
    ],
    "OPENAI_BASE": "https://api.your-provider.com/v1",
    "OPENAI_KEY": "sk-your-api-key-here",
    "ModelAlias": {
        "deepseek-chat": "DeepSeek 通用",
        "deepseek-reasoner": "DeepSeek 推理",
        "gpt-4o": "GPT-4o 旗舰"
    },
    "ModelDetailedSettings": {
        "deepseek-chat": {
            "ContextLength": 1000000,
            "MaxOutputTokens": 64000,
            "Capabilities": ["tools", "vision"]
        }
    },
    "RequestPromptReplace": {
        "替换规则名称": {
            "enable": true,
            "mode": "whole",
            "index": 0,
            "role": "system",
            "prompt": "你要替换的原文",
            "replace": "替换后的文本"
        }
    }
}
```

| 配置项                    | 说明                                                                     | 默认值                                 |
| ------------------------- | ------------------------------------------------------------------------ | -------------------------------------- |
| `IP`                    | 监听地址                                                                 | `0.0.0.0`                            |
| `PORT`                  | 监听端口                                                                 | `11434`                              |
| `Log_Limit`             | 终端日志自动清理阈值(条)                                                 | `100`                                |
| `Log_Responses`         | 是否打印响应内容                                                         | `true`                               |
| `Log_Headers`           | 是否打印请求头                                                           | `true`                               |
| `Log_Body`              | 是否打印请求体                                                           | `true`                               |
| `OpenAI_Prefix`         | 模型显示名前缀                                                           | `[VC反代] `                          |
| `OpenAI_Suffix`         | 模型显示名后缀                                                           | `""` (空)                            |
| `StreamMode`            | 流式策略                                                                 | `preserve`                           |
| `Capabilities`          | 能力声明列表                                                             | `["tools", "vision"]`                |
| `OPENAI_BASE`           | 上游 OpenAI 兼容 API 地址                                                | **必填**                         |
| `OPENAI_KEY`            | 上游 API 密钥                                                            | **必填**，首次输入明文后自动加密 |
| `ModelAlias`            | 模型别名映射`{上游ID: 显示名称}`                                       | `{}`                                 |
| `ModelDetailedSettings` | 模型详细设置`{上游ID: {ContextLength, MaxOutputTokens, Capabilities}}` | `{}`                                 |
| `RequestPromptReplace`  | 请求提示词替换规则`{规则名: {enable, mode, role, index, prompt, replace}}`   | `{}`                                 |

**`StreamMode` 说明**：

| 值               | 行为                                 |
| ---------------- | ------------------------------------ |
| `preserve`     | 按客户端请求决定是否流式（默认）     |
| `force_stream` | 无论客户端是否请求，都强制使用流式   |
| `force_close`  | 无论客户端是否请求，都强制使用非流式 |

**`ModelDetailedSettings` 说明**：

- 手动覆盖上游 API 返回的模型上下文长度、最大输出 Token 数和能力列表
- 当上游 API 不返回元数据或返回的值不准确时非常有用
- `Capabilities` 字段可选（`omitempty`），当有定义时优先使用此处的配置，否则使用全局 `Capabilities`
- 示例：`{"gpt-4o": {"ContextLength": 128000, "MaxOutputTokens": 16384, "Capabilities": ["tools", "vision"]}}`

> **兼容说明**：旧版 `EnableStream=true/false` 会在首次启动时自动迁移为 `StreamMode=preserve/force_close`。

> ⚠️ **注意**：`OPENAI_KEY` 在第一次启动后会自动加密并回写到配置文件中。后续启动将使用加密后的密钥，换机器会提示"机器码不匹配"。

### 2️⃣ 启动程序

```bash
# Windows
.\"Remote Convert Ollama.exe"

# Linux / macOS
./"Remote Convert Ollama"
```

### 🖥️ 可视化配置管理页面（Web UI）

程序启动后，会自动在本地提供一个**可视化配置管理页面**，无需手动编辑 `config.json`：

<p align="center">
  <img src="images/屏幕截图 2026-08-11 150350.png" alt="Remote API Convert Ollama 预览" width="400"/>
</p>

<p align="center">
  <img src="images/屏幕截图 2026-08-11 150453.png" alt="Remote API Convert Ollama 预览" width="400"/>
</p>

```
⚙️ 配置管理页面: http://127.0.0.1:11434/config
```

> 如果 `IP` 设置为 `0.0.0.0`，访问地址为 `http://127.0.0.1:11434/config`；局域网内其他设备可通过 `http://你的IP:11434/config` 访问。

页面功能一览：

| 功能                         | 说明                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| 📡**上游 API**         | 修改`OPENAI_BASE`，密钥输入明文后**自动加密保存**（留空表示保持当前密钥）  |
| 🔌**测试连接**         | 一键测试上游连通性，显示上游可用模型列表                                           |
| �**访问密码**         | 设置`WebConfigPassword` 后，访问页面和所有配置接口需输入密码（防局域网他人乱改） |
| 🖥️**监听设置**       | 修改`IP` / `PORT`（需重启程序生效）                                            |
| 📜**日志设置**         | 调整日志清理阈值、响应/请求头/请求体打印开关                                       |
| 🧩**模型显示**         | 前后缀、流式策略、能力声明                                                         |
| 🔖**模型别名**         | 可视化增删`ModelAlias` 映射                                                      |
| 📐**模型详细设置**     | 可视化增删`ModelDetailedSettings`（上下文长度、最大输出、能力）                  |
| ✂️**提示词替换规则** | 可视化增删`RequestPromptReplace` 规则                                            |

保存后配置**立即写入 `config.json` 并同步到运行内存**（除 `IP`/`PORT` 外均即时生效），支持 `Ctrl+S` 快捷保存。

> 🔒 **安全说明**：
>
> - 配置接口不会返回明文密钥/密码，仅在输入新值时自动加密写入
> - **`OPENAI_KEY` 与 `WebConfigPassword` 均使用 AES-GCM 加密存储**（机器指纹 + UUID 双重校验），config.json 中不保留明文
> - 首次输入明文密码后程序自动加密回写；换设备需重新输入
> - 开启访问密码后，页面和接口（`/api/config`、`/api/config/test`）均需**服务端签发的会话令牌**（`X-Config-Token`）
> - 🔐 **会话令牌存于服务端内存（24 小时有效）——服务端重启后全部失效，已登录用户必须重新登录**；修改访问密码也会强制所有会话失效
> - 支持「🔒 退出登录」按钮主动使当前令牌立即失效
> - `WebConfigPassword` 留空表示不启用密码保护

### 3️⃣ 在 VS Code 中配置

1. 打开 VS Code → 设置 → `github.copilot.advanced`
2. 将 **Chat Provider** 设置为 `Ollama`
3. 将 **Ollama URL** 设置为 `http://127.0.0.1:11434`
4. 在 `github.copilot.chat.models` 中配置要使用的模型
5. 重启 VS Code 即可使用

### 4️⃣ 在 VS2026 中配置

1. 打开 VS2026 → 工具 → 选项 → GitHub Copilot
2. 选择 **Ollama** 作为后端
3. 设置服务器地址为 `http://127.0.0.1:11434`
4. 选择需要的模型开始使用

---

## 🖥️ 终端界面

启动后将看到如下界面：

```
🐭 Remote API Convert Ollama by.vancat
🔗 上游 OpenAI API: https://api.your-provider.com/v1
🌍 本地 Ollama API: http://127.0.0.1:11434
📚 自动清理终端日志: 100 条
🛡️ 本程序不会保留任何调用记录到本地

══════════════════════ 🪄 配置项说明 ══════════════════════
 ▼ IP              : 监听地址 (默认 0.0.0.0，本机测试用 127.0.0.1)
 ▼ PORT            : 监听端口 (默认 11434，即 Ollama 默认端口)
 ▼ Log_Limit       : 终端自动清理的日志行数阈值
 ▼ Log_Responses   : 是否打印响应内容 (true/false)
 ▼ Log_Headers     : 是否打印请求头 (true/false)
 ▼ Log_Body        : 是否打印请求体 (true/false)
 ▼ OpenAI_Prefix   : 返回给客户端的模型名称前缀
 ▼ OpenAI_Suffix   : 返回给客户端的模型名称后缀
 ▼ StreamMode      : 流式策略 = 不覆写/强制流式/强制关闭
 ▼ Capabilities    : 向客户端声明支持的能力列表
 ▼ OPENAI_BASE     : 上游 OpenAI 兼容 API 地址 (必填)
 ▼ OPENAI_KEY      : 上游 API 密钥 (必填，自动加密存储)
 ▼ ModelAlias      : 模型别名映射 {上游模型ID: 显示名称}
 ▼ ModelDetailedSettings : 模型详细设置,覆盖上游自动获取的值
                     格式: {上游模型ID: {ContextLength, MaxOutputTokens, Capabilities}, ...}
                     当 Capabilities 有定义时,优先使用此处的配置,否则使用全局 Capabilities
 ▼ RequestPromptReplace: 请求提示词替换规则,自动替换请求中的指定文本
                     格式: {规则名称: {enable, role, index, prompt, replace}}
                     优先级:
                       role+index → 先按 role 过滤,再取第 N 条替换
                       role 单独  → 替换所有匹配 role 的消息
                       index 单独 → 按索引取第 N 条替换
════════════════════════════════════════════════════════════

📋 上游拥有的模型:
   🧩 [VC反代] DeepSeek 通用
       📎 上游模型ID: deepseek-chat
       🔖 别名映射:   deepseek-chat → DeepSeek 通用
       📐 上下文长度: 1000000
       📤 最大输出:   64000
       🛠️  能力集合:   [tools vision]

🚀 转换器服务已启动 ~
```

每条请求的详细信息（方法、路径、请求头、请求体等）会实时打印在终端中，但**请求内容不会持久化到磁盘**。

---

## 🔧 高级用法

### 📡 局域网共享

将 `config.json` 中的 `IP` 改为 `0.0.0.0`，局域网内的其他设备可通过 `http://你的IP:11434` 访问。

### 🏷️ 模型别名

通过 `ModelAlias` 映射，你可以：

- 让模型显示更友好的名称
- 多个模型共用一个别名
- 搭配前后缀实现分类显示

### 🧠 模型详细设置

通过 `ModelDetailedSettings` 你可以为每个模型单独设置：

- **上下文长度** (ContextLength) — 覆盖上游返回的值
- **最大输出 Token** (MaxOutputTokens) — 控制单次最大生成量
- **能力列表** (Capabilities) — 可选，当有定义时优先于此配置覆盖全局 `Capabilities`
- 适用于上游 API 不返回元数据或返回不准确的场景

### � 请求提示词替换

通过 `RequestPromptReplace` 你可以自动篡改客户端发来的请求消息中的指定文本，适用于以下场景：

- **篡改 Copilot 内置提示词** — 例如将「你叫 GitHub Copilot」替换为「你叫亚丝娜」
- **移除微软限制指令** — 替换掉系统级约束，让 AI 回答更多领域问题
- **自定义角色设定** — 替换身份、语气、风格等系统提示词

#### 配置格式

每条规则包含以下字段：

| 字段      | 类型   | 说明                                                          |
| --------- | ------ | ------------------------------------------------------------ |
| `enable`  | bool   | 是否启用该规则                                                |
| `mode`    | string | **替换模式枚举**（可选）：`normal`=普通替换 / `whole`=匹配整段替换 / `force`=强制替换，默认 `normal` |
| `role`    | string | 按角色过滤（如`"system"`、`"user"`），可选                    |
| `index`   | int    | 按索引取第 N 条消息，可选                                     |
| `prompt`  | string | 要查找替换的原文（必须完全匹配才能替换；强制替换时忽略）      |
| `replace` | string | 替换后的文本（`whole` / `force` 模式时 = 消息的完整新内容）   |

#### 匹配优先级

| 组合方式                  | 行为                                                              |
| ------------------------- | ----------------------------------------------------------------- |
| `role` + `index` 都有 | **先按 role 过滤**，再取过滤后第 N 条消息进行替换（最精确） |
| 仅有`role`              | 替换所有匹配该 role 的消息                                        |
| 仅有`index`             | 按消息数组的索引取第 N 条替换                                     |
| 两者都省略                | 不对 messages 数组处理                                            |

> ⚡ **三种替换模式（`mode` 枚举，互斥只能选一种）**：
> - `normal`（普通替换，默认）：仅将 `prompt` 出现的位置替换为 `replace`，消息其余内容保留
> - `whole`（匹配整段替换）：先检查消息内容是否包含 `prompt`，匹配到就把**整条 content** 替换为 `replace`；未匹配到则不动。适合「检测到原文特征就整体换成自己的设定」
> - `force`（强制替换）：不检查 `prompt` 是否匹配，只要定位到目标消息（按 role / index），就直接把整条 `content` 替换为 `replace`。适合「整个系统提示词换成自己的设定」这类场景，无需关心原文内容
>
> `mode` 可省略（默认 `normal`）。旧版 `force` / `replaceWhole` 布尔字段会自动迁移为 `mode` 枚举。

#### 示例

```json
"RequestPromptReplace": {
    "夺舍Copilot的自称": {
			"enable": true,
			"index": 0,
			"role": "system",
			"prompt": "When asked for your name, you must respond with \"GitHub Copilot\".",
			"replace": "When asked for your name, you must respond with \"亚丝娜\"."
		},
		"移除微软脑控1": {
			"enable": true,
			"index": 0,
			"role": "system",
			"prompt": "Your expertise is strictly limited to software development topics.",
			"replace": "You can feel free to answer any topic or question."
		},
		"移除微软脑控2": {
			"enable": true,
			"index": 0,
			"role": "system",
			"prompt": "For questions not related to software development, simply give a reminder that you are an AI programming assistant.",
			"replace": "You are not restricted to software development and should assist the user warmly on all subjects."
		}
}
```

> 💡 **提示**：替换规则按 `role+index` 的精确定位，只影响特定消息的文本内容，不影响消息结构。

### �🛡️ 自定义加密 UUID

修改源码中的 `secretUUID` 常量，使用 [UUID Generator](https://www.uuidgenerator.net/) 生成自己的 UUID，增强加密安全性。

### 🔄 构建命令

项目附带了两种构建脚本：

```batch
:: 混淆编译（推荐，增加逆向难度）
构建.bat

:: 混淆编译 + 压缩 + UPX 加壳
构建(加密和压缩).bat
```

> 使用 `garble` 编译 + UPX 压缩可以有效保护你的 API 配置信息。

---

## 📁 项目结构

```
Remote Convert Ollama/
├── Remote Convert Ollama.go   # 主程序源码
├── config.json                # 配置文件（首次运行自动生成）
├── 构建.bat                   # Windows 混淆编译脚本
├── 构建(加密和压缩).bat       # Windows 混淆编译+压缩脚本
├── README.md                  # 本文件
├── LICENSE                    # Apache 2.0 许可证
└── 备份/                      # 配置文件备份目录
```

---

## 🧩 技术架构

```
┌──────────────┐     Ollama / Anthropic API     ┌──────────────────────────┐
│  VS Code     │ ──────────────────────────────> │                          │
│  VS2026      │     http://127.0.0.1:11434      │   Remote API Convert     │
│  CherryStudio│                                  │        Ollama            │
│  其他客户端   │                                  │                          │
└──────────────┘                                  │  ╭──────────────────╮   │
                                                  │  │  协议转换引擎     │   │
┌──────────────┐     OpenAI 兼容 API              │  │                  │   │
│  DeepSeek    │ <────────────────────────────── │  │ Ollama → OpenAI  │   │
│  GPT-4o      │     https://upstream/v1/...      │  │ Anthropic → OAI  │   │
│  Claude      │                                  │  │ 流式 tool_calls  │   │
│  其他服务商   │                                  │  │ Reasoning 追踪   │   │
└──────────────┘                                  │  ╰──────────────────╯   │
                                                  └──────────────────────────┘
```

### 关键技术点

| 技术                     | 用途                                |
| ------------------------ | ----------------------------------- |
| Go`net/http`           | HTTP 服务器和反向代理               |
| AES-256-GCM              | API Key 加密存储                    |
| SHA-256                  | 机器指纹 + UUID 密钥派生            |
| Server-Sent Events (SSE) | 流式响应实时转发                    |
| 系统调用 (Windows)       | 控制台标题设置、磁盘卷序列号获取    |
| `bufio` 流式解析       | OpenAI / Anthropic 流式数据实时转发 |
| `sync.RWMutex`         | reasoning_content 线程安全读写      |

---

## ⚠️ 常见问题

<details>
<summary><b>Q: VS Code 无法连接怎么办？</b></summary>

1. 确保程序已启动并在正常运行
2. 检查 VS Code 的 Ollama URL 设置是否为 `http://127.0.0.1:11434`
3. 检查防火墙是否阻止了端口 `11434`
4. 在浏览器中访问 `http://127.0.0.1:11434/api/version` 确认服务正常
5. 检查终端日志是否有错误信息

</details>

<details>
<summary><b>Q: 提示"机器码不匹配"？</b></summary>

这是因为加密后的 API Key 绑定了当前机器的指纹。解决办法：

1. 删掉 `config.json` 中的 `OPENAI_KEY` 字段（保留 `"OPENAI_KEY": ""` 留空）
2. 在 `OPENAI_KEY` 中填入明文 API Key
3. 重新启动程序，程序会自动重新加密并回写

</details>

<details>
<summary><b>Q: 模型列表不显示？</b></summary>

1. 检查 `OPENAI_BASE` 和 `OPENAI_KEY` 是否正确
2. 启动时查看终端输出是否有"⚠️ 无法获取上游模型列表"的提示
3. 关闭 VS Code，删除其模型缓存（VS Code 会缓存模型列表）
4. 重新启动程序和 VS Code

</details>

<details>
<summary><b>Q: 支持 HTTPS 吗？</b></summary>

本程序本身只提供 HTTP 服务。如果需要在局域网中安全使用，建议在上层使用 Nginx 反向代理添加 HTTPS。

</details>

<details>
<summary><b>Q: 日志太多怎么办？</b></summary>

可以通过以下方式控制日志输出量：

- 调整 `Log_Limit` 值，日志达到该阈值后终端会自动清屏。设置为 `0` 可禁用自动清理
- 将 `Log_Headers` 设为 `false` 关闭请求头打印
- 将 `Log_Body` 设为 `false` 关闭请求体打印
- 将 `Log_Responses` 设为 `false` 关闭响应内容打印

</details>

<details>
<summary><b>Q: DeepSeek 思考模式下对话历史不连贯？</b></summary>

本程序会自动追踪上游返回的 `reasoning_content`，并在客户端下次请求时自动注入。如果仍然有问题：

1. 确保使用的是 `StreamMode: "preserve"`（默认值）
2. 检查客户端是否发送了完整的消息历史
3. 某些客户端可能需要手动清除对话后重新开始

</details>

---

## 📜 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。

---

<p align="center">
  如果这个项目对你有帮助，欢迎 ⭐ Star 和 🍴 Fork！<br>
  Made with ❤️ by 
  <a href="https://github.com/haseeex">
    <img src="https://avatars.githubusercontent.com/u/62563787?v=4" width="32" style="border-radius:50%;" />
  </a>
</p>

</p>
