param(
    [switch]$NoVisualizer,
    [switch]$Realtime,
    [switch]$DemoDashboard,
    [switch]$FastReplay
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonExe = "python"
$cppAvailable = (& $PythonExe -c "import correlation_fallback; print('1' if correlation_fallback.cpp_backend_available() else '0')").Trim()
if ($cppAvailable -ne "1") {
    Write-Host "C++ correlation backend is not active. Rebuilding..." -ForegroundColor Yellow
    & powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\build_cpp_backend.ps1" -PythonExe $PythonExe
}

$cppStatus = (& $PythonExe -c "import correlation_fallback; print(correlation_fallback.cpp_backend_available())").Trim()
Write-Host "C++ correlation backend: $cppStatus" -ForegroundColor Green

$argsList = @(
    ".\main.py",
    "--replay",
    "--dem", ".\data\uav_3m_dataset\quick_1200\dem.tif",
    "--nmea", ".\data\uav_3m_dataset\quick_1200\radar_data_1200.nmea",
    "--gt", ".\data\uav_3m_dataset\quick_1200\ground_truth_1200.csv",
    "--lat", "-35.21026955",
    "--lon", "149.08689961",
    "--freq", "10",
    "--speed", "50",
    "--window-size", "64",
    "--step-size", "64",
    "--max-offset", "0",
    "--log-level", "INFO",
    "--quiet-console"
)

if ($NoVisualizer) {
    $argsList += "--no-visualizer"
}

$enableRealtime = $Realtime -or ((-not $NoVisualizer) -and (-not $FastReplay))
$enableDemoDashboard = $DemoDashboard -or ((-not $NoVisualizer) -and (-not $FastReplay))

if ($enableRealtime) {
    $argsList += "--realtime-playback"
}
if ($enableDemoDashboard) {
    $argsList += "--demo-dashboard"
}

& $PythonExe @argsList
