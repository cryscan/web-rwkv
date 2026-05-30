@echo off
REM Build the npm-ready wasm package into .\pkg (ESM, --target web).
REM Usage:
REM   build.cmd                    name: "web-rwkv-wasm"
REM   build.cmd --scope cryscan    name: "@cryscan/web-rwkv-wasm"
REM Any extra args are forwarded to wasm-pack.

@echo ==== building wasm (target web) ====
wasm-pack build --release --target web --out-dir pkg %*
@if errorlevel 1 exit /b 1

@echo ==== done: .\pkg ====
@dir pkg
