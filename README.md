
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

---

## 📖 概述

**Remote API Convert Ollama** 是一个用 Go 编写的轻量级本地反向代理服务器。它监听在本地（或局域网）的 Ollama 兼容端口上，将 **Ollama API** 和 **Anthropic Messages API** 的请求实时转换为 **OpenAI 兼容 API** 请求，并转发到上游服务。

> 💡 **简单来说**：你只需在 VS Code / VS2026 中配置 Ollama 作为 API 提供商，然后指向本程序，就能使用任何 **OpenAI 兼容的 API 服务**（如 DeepSeek、Claude 等）。

---

## 🎯 为什么需要这个工具？

| 问题 | 解决方案 |
|------|---------|
| ❌ VS Code Copilot Chat 只支持 Ollama API 调用本地模型 | ✅ 本程序模拟 Ollama API，实际调用远程 OpenAI 兼容 API |
| ❌ VS2026 仅内置支持 OpenAI + Azure + Anthropic 官方 | ✅ 同时提供 Ollama API + Anthropic Messages API 两种接入方式 |
| ❌ 官方限制多、地区不可用、价格高昂 | ✅ 自由选择任意第三方 OpenAI 兼容服务商 |
| ❌ API Key 明文存储有泄露风险 | ✅ AES-GCM 加密存储，绑定机器指纹 + UUID 双重校验 |

---

## ✨ 核心功能

### 🔄 多协议转换

| 客户端请求 | 转换目标 | 说明 |
|-----------|---------|------|
| `GET /api/version` | → 返回版本信息 | VS Code 探测 Ollama 服务 |
| `GET /api/tags` | → `GET /v1/models` (上游) | 获取模型列表，支持别名与前后缀 |
| `POST /api/show` | → 返回增强模型信息 | 包含上下文窗口、能力声明等 |
| `POST /api/chat` | → `POST /v1/chat/completions` (上游) | Ollama 聊天补全 → OpenAI 格式 |
| `POST /v1/chat/completions` | → `POST /v1/chat/completions` (上游) | 标准 OpenAI 流式/非流式透传 |
| `POST /v1/messages` | → `POST /v1/chat/completions` (上游) | **Anthropic 格式 → OpenAI 格式转换** |
| `GET /v1/models` | → `GET /v1/models` (上游) | 获取上游模型列表 |

### 🖥️ VS Code & VS2026 完美兼容

- ✅ **Capabilities 声明** — 向客户端声明支持 `tools`、`vision` 等能力
- ✅ **模型别名系统** — 通过 `ModelAlias` 配置将上游模型 ID 映射为友好名称
- ✅ **显示名前缀/后缀** — 在客户端看到类似 `[VC反代] 高级智商` 的模型名称
- ✅ **VS2026 思考功能** — 返回 `think: true` 启用推理能力
- ✅ **超大上下文** — 声明 1M tokens 上下文窗口
- ✅ **流式传输 (SSE)** — 支持 `stream: true`，实时输出

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
- **自动获取模型列表**：启动时显示上游所有可用模型及其别名映射

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

# 编译
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
    "OpenAI_Prefix": "[VC反代] ",
    "OpenAI_Suffix": "by vancat",
    "EnableStream": true,
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
    }
}
```

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `IP` | 监听地址 | `0.0.0.0` |
| `PORT` | 监听端口 | `11434` |
| `Log_Limit` | 终端日志自动清理阈值(条) | `100` |
| `OpenAI_Prefix` | 模型显示名前缀 | `[VC反代] ` |
| `OpenAI_Suffix` | 模型显示名后缀 | `(by vancat)` |
| `EnableStream` | 启用流式传输 | `true` |
| `Capabilities` | 能力声明列表 | `["tools", "vision"]` |
| `OPENAI_BASE` | 上游 OpenAI 兼容 API 地址 | **必填** |
| `OPENAI_KEY` | 上游 API 密钥 | **必填**，首次输入明文后自动加密 |
| `ModelAlias` | 模型别名映射 | `{}` |

> ⚠️ **注意**：`OPENAI_KEY` 在第一次启动后会自动加密并回写到配置文件中。后续启动将使用加密后的密钥，换机器会提示"机器码不匹配"。

### 2️⃣ 启动程序

```bash
# Windows
.\"Remote Convert Ollama.exe"

# Linux / macOS
./"Remote Convert Ollama"
```

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
 ▼ IP      : ...
 ...

📋 上游拥有的模型:
   🧩 deepseek-chat → [VC反代] DeepSeek 通用

🚀 转换器服务已启动 ~
```

所有请求的元数据（模型、消息数量、字符数等）会实时显示在终端中，但**请求内容不会持久化到磁盘**。

---

## 🔧 高级用法

### 📡 局域网共享

将 `config.json` 中的 `IP` 改为 `0.0.0.0`，局域网内的其他设备可通过 `http://你的IP:11434` 访问。

### 🏷️ 模型别名

通过 `ModelAlias` 映射，你可以：
- 让模型显示更友好的名称
- 多个模型共用一个别名
- 搭配前后缀实现分类显示

### 🛡️ 自定义加密 UUID

修改源码中的 `secretUUID` 常量，使用 [UUID Generator](https://www.uuidgenerator.net/) 生成自己的 UUID，增强加密安全性。

### 🔄 构建命令

项目附带了 `构建.bat`，运行即可混淆编译：

```batch
garble build -o "Remote Convert Ollama.exe" "Remote Convert Ollama.go"
```

> 使用 `garble` 编译可以增加逆向难度，保护你的 API 配置信息。

---

## 📁 项目结构

```
Remote Convert Ollama/
├── Remote Convert Ollama.go   # 主程序源码
├── config.json                # 配置文件（首次运行自动生成）
├── 构建.bat                   # Windows 构建脚本
└── README.md                  # 本文件
```

---

## 🧩 技术架构

```
┌──────────────┐     Ollama / Anthropic API     ┌──────────────────────┐
│  VS Code     │ ──────────────────────────────> │                      │
│  VS2026      │     http://127.0.0.1:11434      │   Remote API Convert │
│  其他客户端   │                                  │        Ollama        │
└──────────────┘                                  │                      │
                                                  │  ╭──────────────╮   │
┌──────────────┐     OpenAI 兼容 API              │  │ 协议转换引擎  │   │
│  DeepSeek    │ <────────────────────────────── │  │              │   │
│  GPT-4o      │     https://upstream/v1/...      │  │ Ollama→OpenAI │   │
│  Claude      │                                  │  │ Anthropic→OAI │   │
│  其他服务商   │                                  │  ╰──────────────╯   │
└──────────────┘                                  └──────────────────────┘
```

### 关键技术点

| 技术 | 用途 |
|------|------|
| Go `net/http` | HTTP 服务器和反向代理 |
| AES-256-GCM | API Key 加密存储 |
| SHA-256 | 机器指纹 + UUID 密钥派生 |
| Server-Sent Events (SSE) | 流式响应实时转发 |
| 系统调用 (Windows) | 控制台标题设置、磁盘卷序列号获取 |

---

## ⚠️ 常见问题

<details>
<summary><b>Q: VS Code 无法连接怎么办？</b></summary>

1. 确保程序已启动并在正常运行
2. 检查 VS Code 的 Ollama URL 设置是否为 `http://127.0.0.1:11434`
3. 检查防火墙是否阻止了端口 `11434`
4. 在浏览器中访问 `http://127.0.0.1:11434/api/version` 确认服务正常
</details>

<details>
<summary><b>Q: 提示"机器码不匹配"？</b></summary>

这是因为加密后的 API Key 绑定了当前机器的指纹。解决办法：
1. 删掉 `config.json` 中的 `OPENAI_KEY` 字段（保留 `已加密|` 前缀之前的内容）
2. 重新输入明文 API Key 启动程序
3. 程序会自动重新加密
</details>

<details>
<summary><b>Q: 模型列表不显示？</b></summary>

1. 检查 `OPENAI_BASE` 和 `OPENAI_KEY` 是否正确
2. 关闭 VS Code，删除其模型缓存（VS Code 会缓存模型列表）
3. 重新启动程序和 VS Code
</details>

<details>
<summary><b>Q: 支持 HTTPS 吗？</b></summary>

本程序本身只提供 HTTP 服务。如果需要在局域网中安全使用，建议在上层使用 Nginx 反向代理添加 HTTPS。
</details>

<details>
<summary><b>Q: 日志太多怎么办？</b></summary>

调整 `Log_Limit` 值，日志达到该阈值后终端会自动清屏。设置为 `0` 可禁用自动清理。
</details>

---

## 📜 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。

---

<p align="center">
  如果这个项目对你有帮助，欢迎 ⭐ Star 和 🍴 Fork！<br>
  <sub>Made with ❤️ by 我</sub>
</p>
