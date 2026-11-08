@echo off
title FLUX.1 Download Progress Monitor (MB / GB)
cls
echo ======================================================================
echo            FLUX.1 LOCAL GPU MODEL DOWNLOAD PROGRESS MONITOR
echo ======================================================================
echo.

powershell -NoProfile -Command ^
  "$partFile = 'd:\film-argent\backend\models\flux1-schnell-fp8.safetensors.part'; " ^
  "$destFile = 'd:\film-argent\backend\models\flux1-schnell-fp8.safetensors'; " ^
  "$statusFile = 'd:\film-argent\backend\models\download_status.json'; " ^
  "if (Test-Path $statusFile) { " ^
  "  $st = Get-Content $statusFile | ConvertFrom-Json; " ^
  "  Write-Host ('Model Name    : ' + $st.model_name) -ForegroundColor Cyan; " ^
  "  Write-Host ('Status        : ' + $st.status) -ForegroundColor Green; " ^
  "  Write-Host ('Downloaded MB : ' + $st.downloaded_mb + ' MB / ' + $st.total_mb + ' MB (' + $st.percent + '%%)') -ForegroundColor Yellow; " ^
  "  Write-Host ('Downloaded GB : ' + $st.downloaded_gb + ' GB / ' + $st.total_gb + ' GB'); " ^
  "  Write-Host ('Speed         : ' + $st.speed_mb_s + ' MB/s'); " ^
  "} elseif (Test-Path $destFile) { " ^
  "  $item = Get-Item $destFile; " ^
  "  $mb = [math]::Round($item.Length / 1MB, 2); " ^
  "  Write-Host ('[COMPLETED] Downloaded: ' + $mb + ' MB') -ForegroundColor Green; " ^
  "} elseif (Test-Path $partFile) { " ^
  "  $item = Get-Item $partFile; " ^
  "  $mb = [math]::Round($item.Length / 1MB, 2); " ^
  "  Write-Host ('[IN PROGRESS] Downloaded: ' + $mb + ' MB') -ForegroundColor Yellow; " ^
  "} else { " ^
  "  Write-Host '[NOT STARTED] Run download_flux_local.py or download_flux.bat to start downloading.' -ForegroundColor Red; " ^
  "}"

echo.
echo ======================================================================
pause

