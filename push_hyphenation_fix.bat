@echo off
cd /d "%~dp0"
echo Running two-pass transcription tests...
python test_two_pass_transcription.py || goto :fail

echo.
echo Syntax-checking transcription.py...
python -c "import ast; ast.parse(open('transcription.py').read()); print('OK')" || goto :fail

echo.
echo Committing and pushing...
git add -A
git commit -m "Fix pass-2 text reconstruction inserting a space before hyphenated word continuations (operator -led -> operator-led)"
git push
echo.
echo Done. Now deploy on the server:
echo   1. ssh root@178.104.152.111
echo   2. cd /opt/degas-clips ^&^& git pull ^&^& systemctl restart degas
echo   3. systemctl status degas   (confirm "active (running)")
echo.
echo No dependency changes -- venv install not needed this time.
echo Then re-run a clip through Caption Review (a clip that triggers pass 2,
echo i.e. has a low-confidence segment) and check hyphenated words look right.
pause >nul
goto :eof

:fail
echo.
echo TESTS FAILED. Not committing. Read the output above.
pause >nul
