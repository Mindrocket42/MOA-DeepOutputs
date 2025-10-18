@echo off
REM Ensure we start in the repo root no matter how it's launched
cd /d "%~dp0"

REM Activate the conda environment
CALL conda activate moa-deepoutputs

REM Run the single agent module with the specified prompt file
python -m single_agent.single_agent_engine.main --prompt-file single_agent/prompts/my_prompt.txt --verbose

REM Keep window open if launched from Explorer
pause