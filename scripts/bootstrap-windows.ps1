<# ShardGrid idempotent Windows GPU Worker bootstrap (T029).

Safe behavior:
  - Detect-first: records PowerShell, OpenSSH, WSL2, Ubuntu distro, NVIDIA,
    Windows capabilities, Windows-host Conda, and WSL training Conda.
  - Never enables Windows features, starts services, edits firewall rules,
    installs Conda, asks for passwords, or reboots.
  - Windows-host Conda is reported separately and is never treated as the
    WSL2 Linux training runtime.

Exit codes:
  0 healthy
  1 detection failure or degraded state
  2 blocked by manual action(s)
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Json,
    [string]$FindingsDir = $(Join-Path $env:USERPROFILE ".shardgrid\bootstrap"),
    [string]$ExpectedUbuntuDistro = "Ubuntu",
    [int]$MinimumNvidiaDriverMajor = 495
)

$ErrorActionPreference = "Continue"
$mode = if ($Check) { "check" } else { "run" }
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$fileTimestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$manualActions = New-Object System.Collections.Generic.List[string]
$commandsRun = New-Object System.Collections.Generic.List[string]
$health = "healthy"

function Add-ManualAction([string]$message) {
    $script:manualActions.Add($message) | Out-Null
    $script:health = "blocked_manual_action"
}

function Invoke-DetectedCommand([string]$label, [scriptblock]$command) {
    $script:commandsRun.Add($label) | Out-Null
    try {
        & $command 2>&1 | Out-String
    } catch {
        "ERROR: $($_.Exception.Message)"
    }
}

function Get-CommandPath([string]$name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { return $null }
    return $cmd.Source
}

function Get-WindowsCapabilityState([string]$name) {
    try {
        $cap = Get-WindowsCapability -Online -Name $name -ErrorAction Stop 2>$null
        if ($null -eq $cap) { return "not_found" }
        return [string]$cap.State
    } catch {
        return "requires_elevation_to_check"
    }
}

function Get-WindowsOptionalFeatureState([string]$name) {
    try {
        $feature = Get-WindowsOptionalFeature -Online -FeatureName $name -ErrorAction Stop 2>$null
        if ($null -eq $feature) { return "not_found" }
        return [string]$feature.State
    } catch {
        return "requires_elevation_to_check"
    }
}

function Show-Value($value, [string]$fallback = "not_found") {
    if ($null -eq $value -or $value -eq "") { return $fallback }
    return $value
}

function Parse-NvidiaDriverMajor([string]$text) {
    if ($text -match "Driver Version:\s*([0-9]+)") {
        return [int]$Matches[1]
    }
    return $null
}

New-Item -ItemType Directory -Force -Path $FindingsDir | Out-Null

$isAdmin = $false
try {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
} catch {}

$powershellInfo = @{
    edition = $PSVersionTable.PSEdition
    version = $PSVersionTable.PSVersion.ToString()
    elevated = $isAdmin
}

$sshPath = Get-CommandPath "ssh.exe"
$sshdPath = Get-CommandPath "sshd.exe"
$sshVersion = if ($sshPath) { (Invoke-DetectedCommand "ssh -V" { ssh -V }) -replace "\s+$", "" } else { "not_installed" }
$sshdService = Get-Service sshd -ErrorAction SilentlyContinue
$sshServerObserved = [bool]$env:SSH_CONNECTION -or $null -ne $sshdPath -or $null -ne $sshdService

$capabilities = @{
    open_ssh_client = Get-WindowsCapabilityState "OpenSSH.Client~~~~0.0.1.0"
    open_ssh_server = Get-WindowsCapabilityState "OpenSSH.Server~~~~0.0.1.0"
    wsl = Get-WindowsOptionalFeatureState "Microsoft-Windows-Subsystem-Linux"
    virtual_machine_platform = Get-WindowsOptionalFeatureState "VirtualMachinePlatform"
}

if (-not $sshPath -and $capabilities.open_ssh_client -ne "Installed") {
    Add-ManualAction "install OpenSSH Client Windows capability (requires approved administrator action)"
}
if (-not $sshServerObserved -and $capabilities.open_ssh_server -ne "Installed") {
    Add-ManualAction "install OpenSSH Server Windows capability (requires approved administrator action)"
}
if (-not [bool]$env:SSH_CONNECTION -and ($null -eq $sshdService -or $sshdService.Status -ne "Running")) {
    Add-ManualAction "start and enable the sshd service after operator approval (administrator action)"
}

$wslPath = Get-CommandPath "wsl.exe"
$wslStatus = if ($wslPath) { Invoke-DetectedCommand "wsl --status" { wsl --status } } else { "not_installed" }
$wslList = if ($wslPath) { Invoke-DetectedCommand "wsl -l -v" { wsl -l -v } } else { "" }
$ubuntuDistroPresent = $false
$ubuntuDistroVersion2 = $false
if ($wslPath) {
    $plainWslList = $wslList -replace "`0", ""
    $ubuntuDistroPresent = $plainWslList -match "(?im)^\s*\*?\s*$([regex]::Escape($ExpectedUbuntuDistro))\s+"
    $ubuntuDistroVersion2 = $plainWslList -match "(?im)^\s*\*?\s*$([regex]::Escape($ExpectedUbuntuDistro))\s+\S+\s+2\s*$"
}

if (-not $wslPath) {
    Add-ManualAction "enable WSL and VirtualMachinePlatform Windows features (administrator action and reboot may be required)"
}
if (-not $ubuntuDistroPresent) {
    Add-ManualAction "install the Ubuntu WSL distro manually, then rerun this check"
} elseif (-not $ubuntuDistroVersion2) {
    Add-ManualAction "convert the Ubuntu WSL distro to WSL2 manually: wsl --set-version $ExpectedUbuntuDistro 2"
}

$nvidiaPath = Get-CommandPath "nvidia-smi.exe"
$nvidiaSmi = if ($nvidiaPath) { Invoke-DetectedCommand "nvidia-smi" { nvidia-smi } } else { "not_installed" }
$nvidiaDriverMajor = Parse-NvidiaDriverMajor $nvidiaSmi
$nvidiaCompatible = $false
if ($nvidiaDriverMajor -ne $null -and $nvidiaDriverMajor -ge $MinimumNvidiaDriverMajor) {
    $nvidiaCompatible = $true
}
if (-not $nvidiaPath) {
    Add-ManualAction "install an NVIDIA Windows driver that supports CUDA on WSL2, then reboot if the installer requires it"
} elseif (-not $nvidiaCompatible) {
    Add-ManualAction "upgrade the NVIDIA Windows driver to a CUDA-on-WSL2 compatible driver, then reboot if required"
}

$hostCondaPath = Get-CommandPath "conda.exe"
if (-not $hostCondaPath) {
    $hostCondaPath = Get-CommandPath "conda"
}
$hostCondaVersion = if ($hostCondaPath) { (Invoke-DetectedCommand "conda --version" { conda --version }) -replace "\s+$", "" } else { "not_installed" }
$hostCondaEnvs = if ($hostCondaPath) { Invoke-DetectedCommand "conda env list" { conda env list } } else { "" }
if (-not $hostCondaPath) {
    Add-ManualAction "install Windows-host Conda only if host-side tooling needs it; elevated Conda installation is manual"
}

$wslCondaPath = $null
$wslCondaVersion = "not_checked"
$wslCondaEnvs = ""
$wslPythonVersion = "not_checked"
if ($wslPath -and $ubuntuDistroPresent -and $ubuntuDistroVersion2) {
    $wslCondaPath = (Invoke-DetectedCommand "wsl conda path" { wsl -d $ExpectedUbuntuDistro bash -lc "command -v conda || true" }).Trim()
    if ($wslCondaPath) {
        $wslCondaVersion = (Invoke-DetectedCommand "wsl conda --version" { wsl -d $ExpectedUbuntuDistro bash -lc "conda --version" }).Trim()
        $wslCondaEnvs = Invoke-DetectedCommand "wsl conda env list" { wsl -d $ExpectedUbuntuDistro bash -lc "conda env list" }
        $wslPythonVersion = (Invoke-DetectedCommand "wsl python --version" { wsl -d $ExpectedUbuntuDistro bash -lc "python --version 2>&1 || true" }).Trim()
    } else {
        Add-ManualAction "install or select Conda inside the Ubuntu WSL2 training runtime; do not use Windows-host Conda for training"
    }
}

if ($health -eq "healthy") {
    if (-not $nvidiaCompatible -or -not $ubuntuDistroVersion2) {
        $health = "degraded"
    }
}

$findings = [ordered]@{
    script = "bootstrap-windows.ps1"
    mode = $mode
    timestamp = $timestamp
    host = $env:COMPUTERNAME
    health = $health
    powershell = $powershellInfo
    openssh = [ordered]@{
        client_path = $sshPath
        server_path = $sshdPath
        version = $sshVersion
        service_status = if ($sshdService) { [string]$sshdService.Status } elseif ($env:SSH_CONNECTION) { "running_observed_via_ssh" } else { "not_installed" }
    }
    windows_capabilities = $capabilities
    wsl = [ordered]@{
        executable = $wslPath
        status = $wslStatus
        distro_list = $wslList
        expected_ubuntu_distro = $ExpectedUbuntuDistro
        ubuntu_present = $ubuntuDistroPresent
        ubuntu_wsl2 = $ubuntuDistroVersion2
    }
    nvidia = [ordered]@{
        nvidia_smi = $nvidiaPath
        driver_major = $nvidiaDriverMajor
        minimum_driver_major = $MinimumNvidiaDriverMajor
        compatible_for_wsl_cuda = $nvidiaCompatible
        raw = $nvidiaSmi
    }
    windows_host_conda = [ordered]@{
        role = "host_only_not_training_runtime"
        executable = $hostCondaPath
        version = $hostCondaVersion
        active_environment = $env:CONDA_DEFAULT_ENV
        active_prefix = $env:CONDA_PREFIX
        environments_raw = $hostCondaEnvs
    }
    wsl_training_conda = [ordered]@{
        role = "training_runtime"
        distro = $ExpectedUbuntuDistro
        executable = $wslCondaPath
        version = $wslCondaVersion
        python_version = $wslPythonVersion
        environments_raw = $wslCondaEnvs
    }
    manual_actions = @($manualActions)
    commands_run = @($commandsRun)
}

$jsonText = $findings | ConvertTo-Json -Depth 8
$latestPath = Join-Path $FindingsDir "windows-latest.json"
$timestampedPath = Join-Path $FindingsDir "windows-$fileTimestamp.json"
$jsonText | Set-Content -Path $latestPath -Encoding UTF8
$jsonText | Set-Content -Path $timestampedPath -Encoding UTF8

if ($Json) {
    $jsonText
} else {
    Write-Host "ShardGrid Windows bootstrap ($mode mode)"
    Write-Host "host: $env:COMPUTERNAME | timestamp: $timestamp"
    Write-Host "PowerShell: $($powershellInfo.version) edition=$($powershellInfo.edition) elevated=$($powershellInfo.elevated)"
    Write-Host "OpenSSH: client=$(Show-Value $sshPath) server=$(Show-Value $sshdPath) sshd=$($findings.openssh.service_status)"
    Write-Host "WSL: exe=$(Show-Value $wslPath) Ubuntu=$ubuntuDistroPresent WSL2=$ubuntuDistroVersion2"
    Write-Host "NVIDIA: nvidia-smi=$(Show-Value $nvidiaPath) driver_major=$(Show-Value $nvidiaDriverMajor 'unknown') compatible=$nvidiaCompatible"
    Write-Host "Windows host Conda: $(Show-Value $hostCondaPath) $hostCondaVersion (host only)"
    Write-Host "WSL training Conda: $(Show-Value $wslCondaPath) $wslCondaVersion (training runtime)"
    Write-Host "health: $health"
    if ($manualActions.Count -gt 0) {
        Write-Host "manual actions:"
        foreach ($action in $manualActions) {
            Write-Host "  - $action"
        }
    }
    Write-Host "findings: $FindingsDir"
}

if ($health -eq "blocked_manual_action") { exit 2 }
if ($health -eq "healthy") { exit 0 }
exit 1
