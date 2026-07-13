Write-Host "=== Bus 6 devices ==="
Get-PnpDevice | Where-Object { $_.InstanceId -like '*6&1967B3B0*' -and $_.Present -eq $true } | ForEach-Object {
    Write-Host "  $($_.FriendlyName) | $($_.InstanceId) | Status=$($_.Status) | Problem=$($_.Problem)"
}

Write-Host "`n=== Bus 7 devices (1E6F27D5) ==="
Get-PnpDevice | Where-Object { $_.InstanceId -like '*7&1E6F27D5*' -and $_.Present -eq $true } | ForEach-Object {
    Write-Host "  $($_.FriendlyName) | $($_.InstanceId) | Status=$($_.Status) | Problem=$($_.Problem)"
}

Write-Host "`n=== Bus 7 devices (2913CC43) ==="
Get-PnpDevice | Where-Object { $_.InstanceId -like '*7&2913CC43*' -and $_.Present -eq $true } | ForEach-Object {
    Write-Host "  $($_.FriendlyName) | $($_.InstanceId) | Status=$($_.Status) | Problem=$($_.Problem)"
}

Write-Host "`n=== Bus 7 devices (3735ED10) ==="
Get-PnpDevice | Where-Object { $_.InstanceId -like '*7&3735ED10*' -and $_.Present -eq $true } | ForEach-Object {
    Write-Host "  $($_.FriendlyName) | $($_.InstanceId) | Status=$($_.Status) | Problem=$($_.Problem)"
}

Write-Host "`n=== Bus 7 devices (all) ==="
Get-PnpDevice | Where-Object { $_.InstanceId -match '7&[A-F0-9]+' -and $_.Present -eq $true -and ($_.InstanceId -like '*2BC5*' -or $_.InstanceId -like '*0000*0002*') } | ForEach-Object {
    Write-Host "  $($_.FriendlyName) | $($_.InstanceId) | Status=$($_.Status) | Problem=$($_.Problem)"
}
