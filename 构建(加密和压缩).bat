@echo off
chcp 65001 >nul
title 构建 Remote Convert Ollama (加密+压缩)

echo ============================================
echo  [1/3] 停止正在运行的旧版程序...
echo ============================================
taskkill /IM "Remote Convert Ollama.exe" /F >nul 2>&1
if %errorlevel%==0 (
    echo  [OK] 已停止旧进程
) else (
    echo  [SKIP] 无旧进程在运行
)
timeout /t 1 /nobreak >nul

echo ============================================
echo  [2/3] garble 混淆编译...
echo ============================================
garble -tiny -seed=random build -o "Remote Convert Ollama No-Enc.exe" "Remote Convert Ollama.go"
if errorlevel 1 (
    echo  [FAIL] garble 编译失败
    pause
    exit /b 1
)
echo  [OK] 编译完成

echo ============================================
echo  [3/3] UPX 压缩...
echo ============================================
if exist "Remote Convert Ollama.exe" del /f /q "Remote Convert Ollama.exe"
upx --best --lzma "Remote Convert Ollama No-Enc.exe" -o "Remote Convert Ollama.exe"
if errorlevel 1 (
    echo  [FAIL] UPX 压缩失败
    pause
    exit /b 1
)
echo  [OK] 压缩完成

echo ============================================
echo  构建成功: Remote Convert Ollama.exe
echo  如需启动新版,请手动运行 Remote Convert Ollama.exe
echo ============================================
@pause