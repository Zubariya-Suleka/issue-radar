<#
.SYNOPSIS
    Run the issue-radar Docker stack from Windows (D-018).

.DESCRIPTION
    Docker Engine runs inside the dedicated 'issue-radar' WSL distro, whose disk lives on G:.
    WSL stops a distro ~15 s after its last wsl.exe process exits, which would stop Docker
    and the gateway with it. 'up' therefore starts a hidden keep-alive process; 'down'
    removes it, so the distro (and its RAM) is only in use while the stack is wanted.

.EXAMPLE
    .\scripts\stack.ps1 up        # build if needed, start, wait until healthy
    .\scripts\stack.ps1 status
    .\scripts\stack.ps1 logs
    .\scripts\stack.ps1 down
#>
param(
    [Parameter(Mandatory)]
    [ValidateSet("up", "down", "status", "logs")]
    [string]$Command
)

# Native tools (wsl, docker) write progress to stderr. Windows PowerShell 5.1 would turn
# every such line into an error under "Stop", so failures are judged by exit code instead.
$ErrorActionPreference = "Continue"
$Distro = "issue-radar"
$env:WSL_UTF8 = "1"
$RepoWin = Split-Path -Parent $PSScriptRoot
$Repo = (wsl.exe -d $Distro -u root -- wslpath -a ($RepoWin -replace '\\', '/') 2>$null).Trim()

function Invoke-Compose([string[]]$ComposeArgs) {
    wsl.exe -d $Distro -u root --cd $Repo -- docker compose @ComposeArgs 2>&1 |
        ForEach-Object { "$_" } |
        Where-Object { $_ -notmatch "Failed to start the systemd user session" }
    if ($LASTEXITCODE -ne 0) { throw "docker compose $ComposeArgs failed (exit $LASTEXITCODE)" }
}

function Get-KeepAlive {
    Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'" |
        Where-Object { $_.CommandLine -like "*-d $Distro*sleep infinity*" }
}

switch ($Command) {
    "up" {
        if (-not (Get-KeepAlive)) {
            Start-Process wsl.exe -ArgumentList "-d $Distro -u root --exec sleep infinity" -WindowStyle Hidden
        }
        Invoke-Compose @("--progress", "quiet", "up", "-d", "--build")
        # G: is a hard disk: a cold start (after a reboot) takes ~75 s, the very first one
        # (model copied into the volume) ~3 min; a warm restart ~15 s (D-018).
        Write-Host "Waiting for the gateway to load the model (up to 5 min on a cold start)..."
        foreach ($i in 1..100) {
            try {
                $health = Invoke-RestMethod http://localhost:8000/health -TimeoutSec 3
                Write-Host "Gateway up on http://localhost:8000 ($($health.backend))"
                return
            } catch { Start-Sleep -Seconds 3 }
        }
        throw "Gateway did not become healthy in 5 minutes; see: .\scripts\stack.ps1 logs"
    }
    "down" {
        Invoke-Compose @("down")
        Get-KeepAlive | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
        Write-Host "Stack stopped; the distro will shut down by itself in ~15 s."
    }
    "status" {
        Invoke-Compose @("ps")
        $keepAlive = if (Get-KeepAlive) { "running" } else { "not running" }
        Write-Host "Keep-alive: $keepAlive"
    }
    "logs" {
        Invoke-Compose @("logs", "--tail", "50", "gateway")
    }
}
