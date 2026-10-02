@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if not errorlevel 1 goto use_path_node

set "CODEX_NODE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
if exist "%CODEX_NODE%" goto use_codex_node

echo Node.js was not found.
echo Install Node.js 24 or later and add node to PATH, then run this script again.
pause
exit /b 1

:use_path_node
node server.mjs
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:use_codex_node
"%CODEX_NODE%" server.mjs
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
