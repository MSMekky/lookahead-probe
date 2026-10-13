@echo off
rem Full pipeline: data, extraction on every vintage, analysis. Logs go to results\run_all.log.
cd /d "%~dp0\.."
set PY=.venv\Scripts\python.exe
%PY% -m lookahead data      >  results\run_all.log 2>&1 || exit /b 1
%PY% -m lookahead check     >> results\run_all.log 2>&1
%PY% -m lookahead extract   >> results\run_all.log 2>&1 || exit /b 1
%PY% -m lookahead analyze   >> results\run_all.log 2>&1
