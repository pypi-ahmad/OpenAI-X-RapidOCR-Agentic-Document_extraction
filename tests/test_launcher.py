from pathlib import Path


def test_windows_launcher_targets_only_verified_port_listener() -> None:
    script = Path("run_app.cmd").read_text(encoding="utf-8")
    assert "where uv" in script.lower()
    assert "Get-NetTCPConnection -LocalPort 8841 -State Listen" in script
    assert "Sort-Object -Unique" in script
    assert script.count("Get-NetTCPConnection -LocalPort 8841 -State Listen") >= 2
    assert "Stop-Process -Id $listenerPid -Force" in script
    assert "[DateTime]::UtcNow.AddSeconds(5)" in script
    assert "do { $pids = @(Get-NetTCPConnection" in script
    assert "while ($pids.Count -gt 0" in script
    assert "still occupied by PID(s)" in script
    assert "Get-NetTCPConnection" in script
    assert "^|" not in script
    assert "taskkill" not in script.lower()
    assert "uv run streamlit run app.py --server.port 8841" in script
    assert "start /B" not in script
    assert "pythonw" not in script.lower()
    assert "pause" in script.lower()
    assert 'if not "%PORT_EXIT_CODE%"=="0"' in script
