@echo off
:loop
echo Running training script...
python trainer.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo Restarting training script...
goto loop
