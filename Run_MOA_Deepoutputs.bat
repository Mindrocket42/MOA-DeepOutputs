@echo off
REM Ensure we start in the repo root no matter how it's launched
cd /d "%~dp0"

REM Activate the conda environment
CALL conda activate moa-deepoutputs

REM Run the module
python -m deepoutputs_engine.main

REM Keep window open if launched from Explorer
pause
