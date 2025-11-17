# Script pour installer Git sur Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "INSTALLATION DE GIT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier si Git est déjà installé
try {
    $gitVersion = git --version 2>$null
    if ($gitVersion) {
        Write-Host "[OK] Git est deja installe: $gitVersion" -ForegroundColor Green
        Write-Host ""
        Write-Host "Vous pouvez maintenant utiliser:" -ForegroundColor Yellow
        Write-Host "  - push_to_github.bat" -ForegroundColor Yellow
        Write-Host "  - push_to_github.ps1" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Appuyez sur Entree pour quitter"
        exit 0
    }
} catch {
    # Git n'est pas installé, continuer
}

Write-Host "Git n'est pas installe ou n'est pas dans le PATH." -ForegroundColor Yellow
Write-Host ""

# Vérifier si winget est disponible
$wingetAvailable = $false
try {
    $wingetVersion = winget --version 2>$null
    if ($wingetVersion) {
        $wingetAvailable = $true
    }
} catch {
    $wingetAvailable = $false
}

if ($wingetAvailable) {
    Write-Host "Option 1: Installation via winget (Recommandee)" -ForegroundColor Cyan
    Write-Host ""
    $install = Read-Host "Voulez-vous installer Git via winget maintenant? (O/N)"
    if ($install -eq "O" -or $install -eq "o") {
        Write-Host ""
        Write-Host "Installation de Git via winget..." -ForegroundColor Yellow
        Write-Host "ATTENTION: Cette operation necessite des droits administrateur." -ForegroundColor Yellow
        Write-Host ""
        
        # Essayer d'installer avec winget
        try {
            Start-Process winget -ArgumentList "install Git.Git" -Verb RunAs -Wait
            Write-Host ""
            Write-Host "[OK] Installation terminee!" -ForegroundColor Green
            Write-Host ""
            Write-Host "IMPORTANT: Fermez et rouvrez PowerShell pour que Git soit disponible." -ForegroundColor Yellow
            Write-Host ""
        } catch {
            Write-Host "[ERREUR] Echec de l'installation via winget." -ForegroundColor Red
            Write-Host ""
        }
    }
} else {
    Write-Host "winget n'est pas disponible sur ce systeme." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Option 2: Installation manuelle" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Telecharger Git depuis: https://git-scm.com/download/win" -ForegroundColor Yellow
Write-Host "2. Executer le fichier installeur" -ForegroundColor Yellow
Write-Host "3. IMPORTANT: Cocher 'Add Git to PATH' pendant l'installation" -ForegroundColor Yellow
Write-Host "4. Fermer et rouvrir PowerShell apres l'installation" -ForegroundColor Yellow
Write-Host ""

Write-Host "Option 3: Utiliser GitHub Desktop (Interface graphique)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Telecharger depuis: https://desktop.github.com/" -ForegroundColor Yellow
Write-Host ""

Write-Host "Apres l'installation, configurez Git:" -ForegroundColor Cyan
Write-Host '  git config --global user.name "Votre Nom"' -ForegroundColor Gray
Write-Host '  git config --global user.email "votre.email@example.com"' -ForegroundColor Gray
Write-Host ""

Read-Host "Appuyez sur Entree pour quitter"

