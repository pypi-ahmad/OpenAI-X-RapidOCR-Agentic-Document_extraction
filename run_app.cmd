@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv was not found on PATH. Install uv, reopen this console, and retry.
    goto :error
)

echo Checking TCP port 8841...
powershell -NoProfile -Command "$deadline = [DateTime]::UtcNow.AddSeconds(5); do { $pids = @(Get-NetTCPConnection -LocalPort 8841 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique); foreach ($listenerPid in $pids) { $verified = Get-NetTCPConnection -LocalPort 8841 -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $listenerPid }; if ($verified) { Write-Host ('Found TCP port 8841 listener PID ' + $listenerPid); Write-Host ('Stopping verified listener PID ' + $listenerPid + '...'); try { Stop-Process -Id $listenerPid -Force -ErrorAction Stop } catch { $stillOwned = Get-NetTCPConnection -LocalPort 8841 -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $listenerPid }; if ($stillOwned) { throw } } } }; if ($pids.Count -gt 0) { Start-Sleep -Milliseconds 250 } } while ($pids.Count -gt 0 -and [DateTime]::UtcNow -lt $deadline); $remaining = @(Get-NetTCPConnection -LocalPort 8841 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique); if ($remaining.Count -gt 0) { Write-Error ('TCP port 8841 is still occupied by PID(s): ' + ($remaining -join ', ')); exit 1 }"
set "PORT_EXIT_CODE=%ERRORLEVEL%"
if not "%PORT_EXIT_CODE%"=="0" (
    echo ERROR: TCP port 8841 could not be prepared safely.
    goto :error
)

echo Starting Agentic Document Extractor on http://127.0.0.1:8841
echo Streamlit logs will remain visible in this window.
uv run streamlit run app.py --server.port 8841
set "APP_EXIT_CODE=%ERRORLEVEL%"

if not "%APP_EXIT_CODE%"=="0" (
    echo Application exited with error code %APP_EXIT_CODE%.
    goto :pause
)
echo Application exited normally.
goto :pause

:error
set "APP_EXIT_CODE=1"

:pause
echo Press any key to close this window.
pause >nul
exit /b %APP_EXIT_CODE%
