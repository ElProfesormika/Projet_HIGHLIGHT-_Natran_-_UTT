@echo off
echo ========================================
echo PUSH VERS GITHUB - Projet HIGHLIGHT+
echo ========================================
echo.

REM Vérifier si Git est installé
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR: Git n'est pas installe ou n'est pas dans le PATH.
    echo.
    echo Veuillez installer Git depuis: https://git-scm.com/download/win
    echo Ou utiliser: winget install Git.Git
    echo.
    pause
    exit /b 1
)

echo [OK] Git detecte
echo.

REM Vérifier si le dépôt est initialisé
if not exist .git (
    echo Initialisation du depot Git...
    git init
    echo.
)

REM Vérifier le remote
git remote -v >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Configuration du remote GitHub...
    git remote add origin https://github.com/ElProfesormika/Projet_HIGHLIGHT-_Natran_-_UTT.git
    echo.
)

echo Remote configure:
git remote -v
echo.

REM Ajouter tous les fichiers
echo Ajout des fichiers...
git add .
echo.

REM Demander le message de commit
set /p COMMIT_MSG="Entrez le message de commit (ou appuyez sur Entree pour utiliser le message par defaut): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update: Améliorations détection et visualisation GP

echo.
echo Commit avec le message: %COMMIT_MSG%
git commit -m "%COMMIT_MSG%"
echo.

REM Créer la branche main si nécessaire
git branch -M main >nul 2>&1

REM Pousser vers GitHub
echo Poussage vers GitHub...
echo.
echo ATTENTION: Vous devrez peut-etre vous authentifier.
echo Si demande, utilisez un Personal Access Token (PAT) au lieu du mot de passe.
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCES: Code pousse vers GitHub!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ERREUR: Echec du push vers GitHub.
    echo ========================================
    echo.
    echo Verifications:
    echo 1. Avez-vous cree le depot sur GitHub?
    echo 2. Avez-vous les droits d'acces?
    echo 3. Etes-vous authentifie (PAT)?
    echo.
)

pause

