# Run a command inside the cycling-tits conda env with the PATH that `conda activate`
# would set.  Only needed if you do NOT use `conda activate cycling-tits` (on Windows,
# other programs on PATH such as MiKTeX ship DLLs that otherwise shadow the
# environment's own libpng/freetype and make matplotlib crash).
#
#   powershell -ExecutionPolicy Bypass -File tools/run_in_env.ps1 python script.py
param([Parameter(ValueFromRemainingArguments = $true)] $cmd)
$candidates = @("$env:CYCLING_TITS_ENV",
                "$env:USERPROFILE\anaconda3\envs\cycling-tits",
                "$env:USERPROFILE\miniconda3\envs\cycling-tits",
                "$env:USERPROFILE\miniforge3\envs\cycling-tits")
$envp = $candidates | Where-Object { $_ -and (Test-Path "$_\python.exe") } | Select-Object -First 1
if (-not $envp) { Write-Error "cycling-tits environment not found (set CYCLING_TITS_ENV)"; exit 1 }
$env:PATH = "$envp;$envp\Library\mingw-w64\bin;$envp\Library\usr\bin;$envp\Library\bin;$envp\Scripts;" + $env:PATH
$exe = $cmd[0]
if ($exe -eq "python") { $exe = "$envp\python.exe" }
$rest = @(); if ($cmd.Count -gt 1) { $rest = $cmd[1..($cmd.Count - 1)] }
& $exe @rest
exit $LASTEXITCODE
