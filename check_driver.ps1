Write-Host "=== Orbbec Drivers in Driver Store ==="
$result = & pnputil /enum-drivers 2>&1
$found = $false
foreach ($line in $result) {
    if ($line -match "orbbec|Orb|Astra|2BC5") {
        $found = $true
    }
    if ($found) {
        Write-Host $line
    }
    if ($found -and $line -match "Driver date") {
        $found = $false
        Write-Host "---"
    }
}

Write-Host "`n=== Driver Package Files ==="
Get-ChildItem "C:\Windows\System32\DriverStore\FileRepository" -Recurse -Filter "*orbbec*" -ErrorAction SilentlyContinue | Select-Object -First 20 FullName

Get-ChildItem "C:\Windows\System32\DriverStore\FileRepository" -Recurse -Filter "*Orb*" -ErrorAction SilentlyContinue | Select-Object -First 20 FullName

Write-Host "`n=== Orbbec INF files ==="
Get-ChildItem "C:\Windows\INF" -Filter "*orbbec*" -ErrorAction SilentlyContinue
Get-ChildItem "C:\Windows\INF" -Filter "*orb*" -ErrorAction SilentlyContinue
