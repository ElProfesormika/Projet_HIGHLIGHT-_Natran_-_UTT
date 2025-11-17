@echo off
echo Arrêt des processus Python...
taskkill /f /im python.exe >nul 2>&1
timeout /t 2 >nul

echo Nettoyage du cache Python...
for /d /r highlight_plus %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q highlight_plus\*.pyc >nul 2>&1

echo Nettoyage du cache Streamlit...
if exist "%USERPROFILE%\.streamlit\cache" rd /s /q "%USERPROFILE%\.streamlit\cache"

echo Lancement de Streamlit...
python launch_streamlit.py





