# Test coverage report for brogameagent (Windows / MSVC only).
#
# Runs ctest under OpenCppCoverage with --cover_children so every
# brogameagent_*_test.exe (registered via add_test) is instrumented.
# Emits an HTML report at build/coverage/index.html.
#
# Requirements:
#   - OpenCppCoverage installed at C:\Program Files\OpenCppCoverage\
#   - Debug build present in build/ with PDBs
#
# Usage:
#   pwsh scripts/coverage.ps1
#   pwsh scripts/coverage.ps1 -Output build/cov

[CmdletBinding()]
param(
    [string]$Output = 'build/coverage'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$occ = "C:\Program Files\OpenCppCoverage\OpenCppCoverage.exe"
if (-not (Test-Path $occ)) {
    throw "OpenCppCoverage not found at $occ. Install: winget install OpenCppCoverage.OpenCppCoverage"
}

$build = Join-Path $root 'build'
if (-not (Test-Path (Join-Path $build 'tests\Debug'))) {
    throw "Debug tests not found under $build\tests\Debug. Build first: cmake --build build --config Debug"
}

$outAbs = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $root $Output }
if (Test-Path $outAbs) { Remove-Item -Recurse -Force $outAbs }

& $occ `
    --sources "$root\src" `
    --modules brogameagent_ `
    --cover_children `
    --export_type "html:$outAbs" `
    --working_dir $root `
    --quiet `
    -- ctest --test-dir $build -C Debug --output-on-failure

Write-Host ""
Write-Host "Coverage report: $outAbs\index.html"
