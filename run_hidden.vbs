' Hourly publish, launched by Windows Task Scheduler with no console window.
CreateObject("WScript.Shell").Run "wsl.exe -e bash -lc ""python3 /mnt/d/Python/whizard-results/publish_results.py >> /mnt/d/Python/whizard-results/state/publish.log 2>&1""", 0, False
