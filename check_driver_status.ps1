Write-Host "=== Orbbec Depth Devices ==="
$devs = Get-PnpDevice | Where-Object { $_.InstanceId -like '*2BC5*060F*MI_00*' -and $_.Present -eq $true }
foreach ($d in $devs) {
    Write-Host "Name: $($d.FriendlyName)"
    Write-Host "InstanceId: $($d.InstanceId)"
    Write-Host "Status: $($d.Status)"
    Write-Host "Class: $($d.Class)"
    Write-Host "Problem: $($d.Problem)"
    $info = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_DriverInfPath' -ErrorAction SilentlyContinue
    Write-Host "Driver INF: $($info.Data)"
    Write-Host "---"
}
