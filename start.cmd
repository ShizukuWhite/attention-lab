@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if not errorlevel 1 goto use_path_node

set "CODEX_NODE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
if exist "%CODEX_NODE%" goto use_codex_node

echo 未找到 Node.js。
echo 请安装 Node.js 24 或更高版本并将 node 加入 PATH，然后重新运行此脚本。
pause
exit /b 1

:use_path_node
node server.mjs
exit /b %ERRORLEVEL%

:use_codex_node
"%CODEX_NODE%" server.mjs
exit /b %ERRORLEVEL%
