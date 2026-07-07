#garble -literals -tiny -seed=random build -o "Remote Convert Ollama.exe" "Remote Convert Ollama.go"
#upx --best --lzma "Remote Convert Ollama.exe" -o "Remote Convert Ollama UPX.exe"
garble -tiny -seed=random build -o "Remote Convert Ollama No-Enc.exe" "Remote Convert Ollama.go"
upx --best --lzma "Remote Convert Ollama No-Enc.exe" -o "Remote Convert Ollama.exe"