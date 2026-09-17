@echo off
cd /d "%~dp0"
echo Running podcast audio transcription tests...
python test_podcast_audio.py || goto :fail

echo.
echo Re-running two-pass transcription tests (regression check -- shares transcription.py)...
python test_two_pass_transcription.py || goto :fail

echo.
echo Syntax-checking app.py and database.py...
python -c "import ast; [ast.parse(open(f).read()) for f in ['app.py','database.py']]; print('OK')" || goto :fail

echo.
echo Committing and pushing...
git add -A
git commit -m "Add standalone audio transcription endpoint (single-shot + chunked) for Studio's Podcast Page Generator tab, reusing the existing int8 medium model and cross-process lock instead of loading a second Whisper model"
git push
echo.
echo Done. Now deploy on the server:
echo   1. ssh root@178.104.152.111
echo   2. cd /opt/degas-clips ^&^& git pull ^&^& systemctl restart degas
echo   3. systemctl status degas   (confirm "active (running)")
echo.
echo No dependency changes -- venv install not needed this time.
echo Deploy Studio's half of this feature too (push_podcast_tab.bat) before testing end-to-end --
echo the two only work together.
pause >nul
goto :eof

:fail
echo.
echo TESTS FAILED. Not committing. Read the output above.
pause >nul
