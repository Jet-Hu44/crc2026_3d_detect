Write-Host "=== Finding Orbbec Driver Package ==="

# Search for Orbbec driver in driver store
$storeDirs = Get-ChildItem "C:\Windows\System32\DriverStore\FileRepository" -Directory | Where-Object { $_.Name -like "*orbbec*" -or $_.Name -like "*orb*" -or $_.Name -like "*astra*" }
foreach ($dir in $storeDirs) {
    Write-Host "Found: $($dir.FullName)"
    Get-ChildItem $dir.FullName | Select-Object Name
}

# Also search wider
Write-Host "`n=== Searching for *.inf files with orbbec ==="
$infFiles = Get-ChildItem "C:\Windows\System32\DriverStore\FileRepository" -Recurse -Filter "*.inf" -ErrorAction SilentlyContinue | Where-Object { (Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue) -match "Orbbec|2BC5" }
foreach ($f in $infFiles) {
    Write-Host "INF File: $($f.FullName)"
}

Write-Host "`n=== Driver registration details ==="
$result = & pnputil /enum-drivers 2>&1
$inOrbbec = $false
for ($i = 0; $i -lt $result.Count; $i++) {
    if ($result[$i] -match "Orbbec") {
        $inOrbbec = $true
    }
    if ($inOrbbec) {
        Write-Host $result[$i]
        if ($result[$i] -match "^$" -or $i -eq $result.Count - 1) {
            if ($i -gt 0 -and $result[$i-1] -match "Published Name" -or $result[$i] -match "Published Name") {
                # continue
            }
        }
    }
    if ($inOrbbec -and $result[$i] -match "Driver date" -and $result[$i] -notmatch "Orbbec") {
        $inOrbbec = $false
    }
}

# Get the first few lines around Orbbec entry
Write-Host "`n=== Direct search for Orbbec INF ==="
$idx = 0
foreach ($line in $result) {
    if ($line -match "Orbbec") {
        Write-Host "Line $idx : $line"
        for ($j = [Math]::Max(0, $idx-2); $j -le [Math]::Min($result.Count-1, $idx+15); $j++) {
            Write-Host "  $j : $($result[$j])"
        }
        break
    }
    $idx++
}
