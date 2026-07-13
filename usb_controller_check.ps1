Write-Host "=== USB Controllers ==="
Get-PnpDevice -Class 'USB' | Where-Object { $_.FriendlyName -like '*Host Controller*' -or $_.FriendlyName -like '*hci*' -or $_.FriendlyName -like '*Root Hub*' } | ForEach-Object {
    Write-Host "$($_.FriendlyName) | Status: $($_.Status) | $($_.InstanceId)"
}

Write-Host "`n=== All Present Orbbec Devices ==="
Get-PnpDevice | Where-Object { $_.InstanceId -like '*2BC5*' -and $_.Present -eq $true } | ForEach-Object {
    Write-Host "Name: $($_.FriendlyName)"
    Write-Host "  InstanceId: $($_.InstanceId)"
    Write-Host "  Status: $($_.Status)"
    Write-Host "  Class: $($d.Class)"
    Write-Host "  Problem: $($_.Problem)"
    Write-Host "---"
}

Write-Host "`n=== All USB Composite with VID_2BC5 ==="
Get-PnpDevice | Where-Object { $_.InstanceId -like 'USB\VID_2BC5*' -and $_.FriendlyName -like '*Composite*' -and $_.Present -eq $true } | ForEach-Object {
    $parentId = $_.InstanceId
    Write-Host "Composite: $($_.FriendlyName)"
    Write-Host "  ID: $parentId"

    # Extract the hub/port prefix (before MI or the last segment)
    $parentPrefix = ($parentId -replace '\\[^\\]*$', '')
    Write-Host "  Parent prefix: $parentPrefix"

    # Find children
    Get-PnpDevice | Where-Object { $_.InstanceId -like "$parentPrefix*" -and $_.Present -eq $true } | ForEach-Object {
        Write-Host "  -> Child: $($_.FriendlyName) | $($_.InstanceId) | $($_.Status) | $($_.Class)"
    }
    Write-Host "---"
}
