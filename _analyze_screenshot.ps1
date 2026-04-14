Add-Type -AssemblyName System.Drawing
$bmp = [System.Drawing.Bitmap]::FromFile('C:\Users\Machine81\Slazy\desktop_screenshot.png')
Write-Output "Size: $($bmp.Width)x$($bmp.Height)"
$stepX = [math]::Floor($bmp.Width / 4)
$stepY = [math]::Floor($bmp.Height / 3)
for($ry=0; $ry -lt 3; $ry++){
  for($rx=0; $rx -lt 4; $rx++){
    $cx = $rx * $stepX + [math]::Floor($stepX/2)
    $cy = $ry * $stepY + [math]::Floor($stepY/2)
    $c = $bmp.GetPixel($cx,$cy)
    Write-Output "Region[$rx,$ry]($cx,$cy): R=$($c.R) G=$($c.G) B=$($c.B)"
  }
}
$bmp.Dispose()
