param(
    [string]$DemPath = ".\data\fabdem_canberra_wide.tif",
    [string]$NmeaPath = ".\output\case_demo.nmea",
    [string]$GtPath = ".\output\case_demo_gt.csv",
    [double]$Lat = -35.36,
    [double]$Lon = 149.05,
    [int]$Trajectory = 1,
    [double]$FreqHz = 5.0
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $NmeaPath) | Out-Null

Write-Host "[1/2] Generating NMEA GPGGA stream into $NmeaPath" -ForegroundColor Cyan
python .\sim_generator.py `
  --dem $DemPath `
  --lat $Lat `
  --lon $Lon `
  --trajectory $Trajectory `
  --freq $FreqHz `
  --output file `
  --out-nmea $NmeaPath `
  --out-csv $GtPath

Write-Host "[2/2] Starting TERRAIN NAVIGATOR with explicit DEM + NMEA input" -ForegroundColor Cyan
python .\main.py `
  --replay `
  --dem $DemPath `
  --nmea $NmeaPath `
  --gt $GtPath `
  --lat $Lat `
  --lon $Lon
