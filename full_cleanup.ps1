Write-Host "=== Cleaning ALL Orbbec related devices ==="

# Remove ALL Orbbec devices - present and phantom
$allOrb = Get-PnpDevice | Where-Object { $_.InstanceId -like '*2BC5*' }
foreach ($d in $allOrb) {
    Write-Host "Removing: $($d.FriendlyName) ($($d.InstanceId))"
    $result = & pnputil /remove-device $d.InstanceId 2>&1
    Write-Host "  Result: $result"
}

Write-Host "`n=== Removing the failed USB device ==="
$failed = Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_0000&PID_0002*' }
foreach ($d in $failed) {
    Write-Host "Removing failed: $($d.FriendlyName) ($($d.InstanceId))"
    $result = & pnputil /remove-device $d.InstanceId 2>&1
    Write-Host "  Result: $result"
}

Write-Host "`n=== Adding Orbbec driver to store ==="
$infPath = "C:\Windows\System32\DriverStore\FileRepository\obdrv4.inf_amd64_173272cad4a99215\obdrv4.inf"
$result = & pnputil /add-driver $infPath 2>&1
Write-Host "Result: $result"

Write-Host "`n=== Scanning for hardware changes ==="
$result = & pnputil /scan-devices 2>&1
Write-Host "Result: $result"

Write-Host "`n=== Done! Now unplug and replug the camera. ==="
