Write-Host "=== Current Orbbec Depth Devices ==="
$devs = Get-PnpDevice | Where-Object { $_.InstanceId -like '*2BC5*060F*MI_00*' -and $_.Present -eq $true }
foreach ($d in $devs) {
    Write-Host "Name: $($d.FriendlyName)"
    Write-Host "InstanceId: $($d.InstanceId)"
    Write-Host "Status: $($d.Status)"
    Write-Host "Class: $($d.Class)"
    Write-Host "Problem: $($d.Problem)"

    # Check driver info
    $info = Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName 'DEVPKEY_Device_DriverInfPath' -ErrorAction SilentlyContinue
    Write-Host "Driver INF: $($info.Data)"
    Write-Host "---"
}

Write-Host "`n=== Attempting to reinstall Orbbec driver ==="
# Uninstall current WinUSB driver first
$depthDev = $devs | Where-Object { $_.InstanceId -like '*MI_00*' }
if ($depthDev) {
    Write-Host "Found depth device, trying to reinstall Orbbec driver..."

    # Use devcon or pnputil to install the correct driver
    $infPath = "C:\Windows\System32\DriverStore\FileRepository\obdrv4.inf_amd64_173272cad4a99215\obdrv4.inf"

    if (Test-Path $infPath) {
        Write-Host "INF found at: $infPath"
        Write-Host "Hardware IDs in INF include: USB\VID_2BC5&PID_060F&MI_00"

        # Try pnputil to add the driver package and install it
        $result = & pnputil /add-driver $infPath /install 2>&1
        Write-Host "pnputil result: $result"
    } else {
        Write-Host "ERROR: INF not found!"
    }
} else {
    Write-Host "No current MI_00 depth device found"
}

Write-Host "`n=== All Orbbec devices after ==="
Get-PnpDevice | Where-Object { $_.InstanceId -like '*2BC5*' } | ForEach-Object {
    Write-Host "$($_.FriendlyName) | $($_.InstanceId) | $($_.Status) | $($_.Problem)"
}
