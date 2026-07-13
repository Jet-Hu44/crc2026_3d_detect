# Check all USB devices with VID 2BC5
Write-Host "=== Orbbec USB Devices ==="
$devs = Get-PnpDevice | Where-Object { $_.InstanceId -like '*2BC5*' }
if ($devs) {
    foreach ($d in $devs) {
        Write-Host "Name: $($d.FriendlyName)"
        Write-Host "Status: $($d.Status)"
        Write-Host "Problem: $($d.Problem)"
        Write-Host "InstanceId: $($d.InstanceId)"
        Write-Host "Class: $($d.Class)"
        Write-Host "---"
    }
} else {
    Write-Host "No device with VID 2BC5 found"
}

Write-Host "`n=== All Image Devices (Cameras) ==="
$cam = Get-PnpDevice -Class 'Image' -ErrorAction SilentlyContinue
foreach ($c in $cam) {
    Write-Host "Name: $($c.FriendlyName), Status: $($c.Status), Id: $($c.InstanceId)"
}

Write-Host "`n=== Unknown/Error Devices ==="
$unk = Get-PnpDevice | Where-Object { $_.Status -eq 'Unknown' -or $_.Status -eq 'Error' }
foreach ($d in $unk) {
    Write-Host "Name: $($d.FriendlyName), Status: $($d.Status), Id: $($d.InstanceId)"
}
