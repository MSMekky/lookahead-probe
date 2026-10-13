@echo off
rem One-time setup on Windows with an NVIDIA GPU (driver with CUDA 11.8 or newer).
rem --isolated ignores user pip.ini files that point at unreachable indexes.
cd /d "%~dp0\.."
if not exist .venv\Scripts\python.exe python -m venv .venv
.venv\Scripts\python.exe -m pip --isolated install --upgrade pip
.venv\Scripts\python.exe -m pip --isolated install torch --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python.exe -m pip --isolated install -e .[gpu,dev]
