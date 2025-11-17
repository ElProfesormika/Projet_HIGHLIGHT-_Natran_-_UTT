# Script PowerShell pour pousser le projet vers GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PUSH VERS GITHUB - Projet HIGHLIGHT+" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier si Git est installé
try {
    $gitVersion = git --version
    Write-Host "[OK] Git detecte: $gitVersion" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "ERREUR: Git n'est pas installe ou n'est pas dans le PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "Veuillez installer Git depuis: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Ou utiliser: winget install Git.Git" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Appuyez sur Entree pour quitter"
    exit 1
}

# Vérifier si le dépôt est initialisé
if (-not (Test-Path .git)) {
    Write-Host "Initialisation du depot Git..." -ForegroundColor Yellow
    git init
    Write-Host ""
}

# Vérifier le remote
$remoteExists = git remote -v 2>$null
if (-not $remoteExists) {
    Write-Host "Configuration du remote GitHub..." -ForegroundColor Yellow
    git remote add origin https://github.com/ElProfesormika/Projet_HIGHLIGHT-_Natran_-_UTT.git
    Write-Host ""
}

Write-Host "Remote configure:" -ForegroundColor Cyan
git remote -v
Write-Host ""

# Ajouter tous les fichiers
Write-Host "Ajout des fichiers..." -ForegroundColor Yellow
git add .
Write-Host ""

# Demander le message de commit
$defaultMsg = "Update: Améliorations détection et visualisation GP"
$commitMsg = Read-Host "Entrez le message de commit (ou appuyez sur Entree pour utiliser: '$defaultMsg')"
if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = $defaultMsg
}

Write-Host ""
Write-Host "Commit avec le message: $commitMsg" -ForegroundColor Yellow
git commit -m $commitMsg
Write-Host ""

# Créer la branche main si nécessaire
git branch -M main 2>$null | Out-Null

# Pousser vers GitHub
Write-Host "Poussage vers GitHub..." -ForegroundColor Yellow
Write-Host ""
Write-Host "ATTENTION: Vous devrez peut-etre vous authentifier." -ForegroundColor Yellow
Write-Host "Si demande, utilisez un Personal Access Token (PAT) au lieu du mot de passe." -ForegroundColor Yellow
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCES: Code pousse vers GitHub!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERREUR: Echec du push vers GitHub." -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Verifications:" -ForegroundColor Yellow
    Write-Host "1. Avez-vous cree le depot sur GitHub?" -ForegroundColor Yellow
    Write-Host "2. Avez-vous les droits d'acces?" -ForegroundColor Yellow
    Write-Host "3. Etes-vous authentifie (PAT)?" -ForegroundColor Yellow
    Write-Host ""
}

Read-Host "Appuyez sur Entree pour quitter"

