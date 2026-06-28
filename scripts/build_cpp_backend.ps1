param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CoreDir = Join-Path $ProjectRoot "terrain_nav_core"
$BuildDir = Join-Path $CoreDir "build"

& $PythonExe -m pip install --upgrade pybind11 cmake

$pybindCmakeDir = (& $PythonExe -c "import pybind11; print(pybind11.get_cmake_dir())").Trim()
& cmake -S $CoreDir -B $BuildDir -Dpybind11_DIR="$pybindCmakeDir" -DPython_EXECUTABLE="$((Get-Command $PythonExe).Source)"
& cmake --build $BuildDir --config Release

$builtModule = Get-ChildItem -Path $CoreDir -Recurse -Filter "_terrain_nav_core*.pyd" |
    Where-Object { $_.FullName -notlike "*\build\*" } |
    Select-Object -First 1
if ($null -ne $builtModule -and $builtModule.DirectoryName -ne $CoreDir) {
    Copy-Item -LiteralPath $builtModule.FullName -Destination $CoreDir -Force
}
& $PythonExe -c "import correlation_fallback; print('cpp_backend_available=', correlation_fallback.cpp_backend_available())"
