# Renders text pages with GDI+, which shapes Indic scripts correctly (conjuncts,
# pre-base vowel signs, reph). Input: a JSON file with a list of {path, text} objects.
param([string]$Spec, [int]$Width = 1654, [int]$Height = 2339, [int]$FontPx = 34, [int]$Margin = 120)
Add-Type -AssemblyName System.Drawing
$items = Get-Content -Raw -Encoding UTF8 $Spec | ConvertFrom-Json
foreach ($item in $items) {
    $bmp = New-Object System.Drawing.Bitmap $Width, $Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear([System.Drawing.Color]::White)
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $font = New-Object System.Drawing.Font("Nirmala UI", $FontPx, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $rect = New-Object System.Drawing.RectangleF $Margin, $Margin, ($Width - 2 * $Margin), ($Height - 2 * $Margin)
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Trimming = [System.Drawing.StringTrimming]::Word
    $g.DrawString($item.text, $font, [System.Drawing.Brushes]::Black, $rect, $fmt)
    $g.Dispose()
    $bmp.Save($item.path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}
"rendered $($items.Count) pages"
