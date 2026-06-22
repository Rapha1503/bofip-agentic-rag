param(
    [string]$RunDir,
    [string]$ConversationUrl = "",
    [int]$MaxWaitMs = 600000,
    [int]$MinChars = 1200
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    throw "RunDir is required."
}

$resolvedRunDir = (Resolve-Path -LiteralPath $RunDir -ErrorAction Stop).Path
$reviewDir = Join-Path $resolvedRunDir "chatgpt-review"
$contextFile = Join-Path $reviewDir "context.md"
$promptsFile = Join-Path $reviewDir "prompts.md"

if (-not (Test-Path -LiteralPath $contextFile -PathType Leaf)) {
    throw "Required ChatGPT review context not found: $contextFile"
}

if (-not (Test-Path -LiteralPath $promptsFile -PathType Leaf)) {
    throw "Required ChatGPT review prompts not found: $promptsFile"
}

$bridgeScript = "C:\Users\rapha\Codex-20x\scripts\chatgpt-debate.ps1"
if (-not (Test-Path -LiteralPath $bridgeScript -PathType Leaf)) {
    throw "Codex-20x ChatGPT bridge not found: $bridgeScript"
}

Write-Host "ChatGPT review run dir: $resolvedRunDir"
Write-Host "ChatGPT review path: $reviewDir"
Write-Host "ChatGPT bridge: $bridgeScript"

& powershell -NoProfile -ExecutionPolicy Bypass -File $bridgeScript `
    -ConversationUrl $ConversationUrl `
    -ContextFile $contextFile `
    -PromptsFile $promptsFile `
    -Title "bofip-rag-review" `
    -MaxWaitMs $MaxWaitMs `
    -MinChars $MinChars `
    -RequireSections "Verdict","Remaining blockers","Recommended next fixes","Minimal validation set" `
    -RequireEndMarker "END_OF_RESPONSE"

$bridgeExitCode = $LASTEXITCODE
if ($bridgeExitCode -ne 0) {
    Write-Error "ChatGPT bridge failed with exit code $bridgeExitCode for review path: $reviewDir"
    exit $bridgeExitCode
}

Write-Host "ChatGPT bridge completed for review path: $reviewDir"
