# Installation de Git sur Windows

## Option 1 : Installation via winget (Recommandé - Rapide)

Ouvrez PowerShell en tant qu'administrateur et exécutez :

```powershell
winget install Git.Git
```

## Option 2 : Installation manuelle

1. **Télécharger Git** :
   - Aller sur : https://git-scm.com/download/win
   - Télécharger la version pour Windows (64-bit)

2. **Installer Git** :
   - Exécuter le fichier téléchargé
   - **IMPORTANT** : Pendant l'installation, cocher l'option :
     - "Add Git to PATH" ou "Ajouter Git au PATH"
   - Suivre les étapes par défaut

3. **Vérifier l'installation** :
   - Fermer et rouvrir PowerShell
   - Exécuter : `git --version`

## Option 3 : Installation via Chocolatey

Si vous avez Chocolatey installé :

```powershell
choco install git
```

## Après l'installation

1. **Fermer et rouvrir PowerShell** (important pour que le PATH soit mis à jour)

2. **Configurer Git** (première utilisation) :
   ```powershell
   git config --global user.name "Votre Nom"
   git config --global user.email "votre.email@example.com"
   ```

3. **Vérifier que Git fonctionne** :
   ```powershell
   git --version
   ```

## Alternative : Utiliser GitHub Desktop

Si vous préférez une interface graphique :

1. Télécharger GitHub Desktop : https://desktop.github.com/
2. Installer et se connecter avec votre compte GitHub
3. Ajouter le dépôt local
4. Faire le commit et push via l'interface

## Vérification rapide

Après installation, testez dans PowerShell :

```powershell
git --version
```

Si cela fonctionne, vous pouvez utiliser les scripts `push_to_github.bat` ou `push_to_github.ps1`.

