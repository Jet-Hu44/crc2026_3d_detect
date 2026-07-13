# Switch depth sensor from WinUSB back to Orbbec driver
# Target device: USB\VID_2BC5&PID_060F&MI_00\7&1E6F27D5&0&0000

$deviceId = "USB\VID_2BC5&PID_060F&MI_00\7&1E6F27D5&0&0000"
$infPath = "C:\Windows\System32\DriverStore\FileRepository\obdrv4.inf_amd64_173272cad4a99215\obdrv4.inf"

Write-Host "=== Step 1: Uninstall current WinUSB driver ==="
$result = & pnputil /remove-device $deviceId 2>&1
Write-Host "Result: $result"

Write-Host "`n=== Step 2: Scan for hardware changes ==="
$result = & pnputil /scan-devices 2>&1
Write-Host "Result: $result"

Start-Sleep -Seconds 3

Write-Host "`n=== Step 3: Check if device re-appeared with correct driver ==="
$devs = Get-PnpDevice | Where-Object { $_.InstanceId -like '*2BC5*060F*MI_00*' -and $_.Present -eq $true }
foreach ($d in $devs) {
    Write-Host "Name: $($d.FriendlyName)"
    Write-Host "InstanceId: $($d.InstanceId)"
    Write-Host "Status: $($d.Status)"
    Write-Host "Class: $($d.Class)"
    $info = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_DriverInfPath' -ErrorAction SilentlyContinue
    Write-Host "Driver INF: $($info.Data)"
    Write-Host "---"
}
