$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Assert-Success($action) { if ($LASTEXITCODE -ne 0) { throw "$action failed." } }
$profileJson = gh api user
Assert-Success 'GitHub authentication'
$profile = $profileJson | ConvertFrom-Json
$login = $profile.login
$repo = "$login/mlr-ex08"
$url = "https://$login.github.io/mlr-ex08/"
$existing = gh repo view $repo --json nameWithOwner,isPrivate 2>$null
if ($LASTEXITCODE -eq 0) {
    $metadata = $existing | ConvertFrom-Json
    if ($metadata.nameWithOwner -ne $repo -or $metadata.isPrivate) { throw 'Expected a public repository owned by your GitHub account.' }
    Write-Host "Resuming publication to $repo"
} else {
    gh repo create $repo --public --description 'Live browser leaf classifier with ONNX Runtime Web'
    Assert-Success 'Repository creation'
}
$readme = Get-Content -Raw README.md
$readme = $readme.Replace('Deployment is pending: this environment blocks GitHub CLI network access. After publishing, the URL is `https://<your-github-login>.github.io/mlr-ex08/`.', "Public page: [$url]($url).")
Set-Content -Path README.md -Value $readme -Encoding utf8
if (!(Test-Path .git)) { git init -b main; Assert-Success 'Git initialization' }
# Set identity only for this repository, and preserve any existing identity.
$authorName = git config user.name
if (!$authorName) { git config --local user.name $login; Assert-Success 'Git author name' }
$authorEmail = git config user.email
if (!$authorEmail) {
    git config --local user.email "$($profile.id)+$login@users.noreply.github.com"
    Assert-Success 'Git author email'
}
git add index.html README.md model_info.json leaf_model.onnx check_image.jpg check_expected.json .nojekyll .gitignore publish.ps1 serve.ps1
Assert-Success 'Staging'
git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    git commit -m 'Build live leaf classifier with camera, photo fallback and browser QR'
    Assert-Success 'Commit'
} elseif ($LASTEXITCODE -ne 0) { throw 'Could not inspect staged changes.' }
$origin = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0) {
    git remote add origin "https://github.com/$repo.git"
    Assert-Success 'Remote setup'
} elseif ($origin -ne "https://github.com/$repo.git" -and $origin -ne "git@github.com:$repo.git") {
    throw "Origin points to a different repository: $origin"
}
git push -u origin main
Assert-Success 'Push'
$pages = gh api "repos/$repo/pages" 2>$null
if ($LASTEXITCODE -eq 0) {
    gh api --method PUT "repos/$repo/pages" -f 'source[branch]=main' -f 'source[path]=/'
} else {
    gh api --method POST "repos/$repo/pages" -f 'source[branch]=main' -f 'source[path]=/'
}
Assert-Success 'Enabling Pages'
gh api --method PUT "repos/$repo/collaborators/MLRteacher" -f permission=push
Assert-Success 'Collaborator invitation'
$online = $false
for ($attempt=0; $attempt -lt 60; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing
        if ($response.StatusCode -eq 200 -and $response.Content.Contains('Meet the leaf')) { $online=$true; break }
    } catch { }
    Start-Sleep -Seconds 10
}
if (!$online) { throw "Pages has not become available yet. Check gh api repos/$repo/pages/builds/latest" }
Write-Host "Live page: $url"
Write-Host "Next: open ${url}?selftest=1 and confirm PASS; scan the footer QR and confirm it opens $url"
