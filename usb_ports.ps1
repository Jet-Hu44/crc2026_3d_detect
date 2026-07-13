Write-Host "=== USB Host Controllers ==="
Get-PnpDevice | Where-Object { $_.Class -eq 'USB' -and $_.FriendlyName -like '*Host Controller*' } | ForEach-Object {
    Write-Host "Controller: $($_.FriendlyName)"
    Write-Host "  InstanceId: $($_.InstanceId)"

    # Get the bus number from the instance ID
    $locProp = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName 'DEVPKEY_Device_LocationInfo' -ErrorAction SilentlyContinue
    Write-Host "  Location: $($locProp.Data)"

    $addrProp = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName 'DEVPKEY_Device_Address' -ErrorAction SilentlyContinue
    Write-Host "  Address: $($addrProp.Data)"
    Write-Host "---"
}

Write-Host "`n=== USB Root Hubs ==="
Get-PnpDevice | Where-Object { $_.Class -eq 'USB' -and $_.FriendlyName -like '*Root Hub*' } | ForEach-Object {
    Write-Host "$($_.FriendlyName) | $($_.InstanceId)"
    $locProp = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName 'DEVPKEY_Device_LocationInfo' -ErrorAction SilentlyContinue
    Write-Host "  Location: $($locProp.Data)"
}

Write-Host "`n=== Current Orbbec Location Paths ==="
$devs = Get-PnpDevice | Where-Object { $_.InstanceId -like '*2BC5*' }
foreach ($d in $devs) {
    Write-Host "Device: $($d.FriendlyName)"
    Write-Host "  InstanceId: $($d.InstanceId)"
    $locProp = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_LocationPaths' -ErrorAction SilentlyContinue
    Write-Host "  LocationPaths: $($locProp.Data -join ' | ')"
    Write-Host "---"
}
