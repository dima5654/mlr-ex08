$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($PSScriptRoot)
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 8000)
$types = @{'.html'='text/html; charset=utf-8'; '.json'='application/json'; '.jpg'='image/jpeg'; '.onnx'='application/octet-stream'; '.wasm'='application/wasm'}
try {
    $listener.Start()
    Write-Host 'Leaf Lens: http://localhost:8000/'
    Write-Host 'Keep this window open. Press Ctrl+C to stop. Use the HTTPS deployment for iPhone camera access.'
    while ($true) {
        $client = $listener.AcceptTcpClient()
        try {
            $client.ReceiveTimeout = 5000
            $stream = $client.GetStream()
            $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::ASCII, $false, 1024, $true)
            $request = $reader.ReadLine()
            if (!$request) { continue }
            while ($reader.ReadLine()) { }
            $parts = $request.Split(' ')
            $relative = [Uri]::UnescapeDataString(($parts[1] -split '\?')[0]).TrimStart('/')
            if (!$relative) { $relative = 'index.html' }
            $path = [IO.Path]::GetFullPath((Join-Path $root $relative))
            $allowed = @('index.html','leaf_model.onnx','model_info.json','check_image.jpg','check_expected.json')
            $ok = $parts[0] -eq 'GET' -and $path.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -and $relative -in $allowed -and [IO.File]::Exists($path)
            if ($ok) {
                $body = [IO.File]::ReadAllBytes($path)
                $code = '200 OK'
                $type = $types[[IO.Path]::GetExtension($path)]
            } else {
                $body = [Text.Encoding]::UTF8.GetBytes('Not found')
                $code = '404 Not Found'
                $type = 'text/plain'
            }
            $header = [Text.Encoding]::ASCII.GetBytes("HTTP/1.1 $code`r`nContent-Type: $type`r`nContent-Length: $($body.Length)`r`nConnection: close`r`nX-Content-Type-Options: nosniff`r`n`r`n")
            $stream.Write($header, 0, $header.Length)
            $stream.Write($body, 0, $body.Length)
        } catch {
            Write-Warning $_.Exception.Message
        } finally {
            $client.Dispose()
        }
    }
} finally {
    $listener.Stop()
}
