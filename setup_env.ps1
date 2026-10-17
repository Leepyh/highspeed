<#
.SYNOPSIS
    Create and configure the training environment for the highspeed project.

.DESCRIPTION
    Steps:
      1. Create the project virtual environment .venv
      2. Upgrade pip / setuptools / wheel
      3. Install PyTorch (CUDA 12.8 build, required for RTX 50-series / Blackwell sm_120)
      4. Install the remaining dependencies
      5. Verify torch sees the GPU and the project modules import correctly

    NOTE: this file is intentionally ASCII-only so it runs correctly under both
    Windows PowerShell 5.1 and PowerShell 7+ (5.1 reads .ps1 as ANSI/GBK).

.PARAMETER PythonExe
    Python interpreter used to create the venv. Default: D:\python\python.exe

.PARAMETER CpuOnly
    Install the CPU-only torch build (about 200MB, much slower). Default is CUDA 12.8.

.PARAMETER TorchVersion
    Version of torch to install. Default 2.9.1 (same cu128 index, supports Blackwell sm_120).
    Pass an empty string to let pip pick the newest available.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_env.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_env.ps1 -CpuOnly

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_env.ps1 -TorchVersion ''
#>

[CmdletBinding()]
param(
    [string]$PythonExe = 'D:\python\python.exe',
    [switch]$CpuOnly,
    [string]$TorchVersion = '2.9.1'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$VenvDir     = Join-Path $ProjectRoot '.venv'
$VenvPython  = Join-Path $VenvDir 'Scripts\python.exe'

function Write-Step($msg) {
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan
}
function Write-Ok($msg)    { Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------- preflight
Write-Step 'Step 0/5  Preflight checks'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python interpreter not found: $PythonExe  (use -PythonExe to override)"
}
$pyVersion = & $PythonExe -c 'import sys; print(sys.version.split()[0])'
Write-Ok "Python: $PythonExe  (version $pyVersion)"

$hasGpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($gpuName) { $hasGpu = $true; Write-Ok "GPU detected: $gpuName" }
}
if (-not $hasGpu) {
    Write-Warn2 'No NVIDIA GPU detected. Consider -CpuOnly to save download size.'
}

# ---------------------------------------------------------------- venv
Write-Step 'Step 1/5  Create virtual environment .venv'

if (Test-Path -LiteralPath $VenvDir) {
    Write-Host "  Removing existing $VenvDir ..." -ForegroundColor Yellow
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}
& $PythonExe -m venv $VenvDir
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "venv creation failed: $VenvPython not found"
}
Write-Ok "Created $VenvDir"

# ---------------------------------------------------------------- pip
Write-Step 'Step 2/5  Upgrade pip / setuptools / wheel'
& $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
Write-Ok (& $VenvPython -m pip --version)

# ---------------------------------------------------------------- torch
$torchSpec = 'torch'
if ($TorchVersion -ne '') { $torchSpec = "torch==$TorchVersion" }

if ($CpuOnly) {
    Write-Step 'Step 3/5  Install PyTorch (CPU build)'
    & $VenvPython -m pip install $torchSpec --quiet
} else {
    Write-Step "Step 3/5  Install PyTorch ($torchSpec, CUDA 12.8, about 2.5-3GB, please wait)"
    Write-Host '  index: https://download.pytorch.org/whl/cu128' -ForegroundColor DarkGray
    & $VenvPython -m pip install $torchSpec --index-url https://download.pytorch.org/whl/cu128
}
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
Write-Ok 'PyTorch installed'

# ---------------------------------------------------------------- others
Write-Step 'Step 4/5  Install remaining dependencies'
& $VenvPython -m pip install numpy matplotlib gymnasium pandas seaborn --quiet
if ($LASTEXITCODE -ne 0) { throw 'dependency installation failed' }
Write-Ok 'numpy / matplotlib / gymnasium / pandas / seaborn installed'

# ---------------------------------------------------------------- verify
Write-Step 'Step 5/5  Verify installation'

$verifyScript = Join-Path $ProjectRoot ('.verify_env_{0}.py' -f (Get-Random))
$check = @'
import sys
print("Python      :", sys.version.split()[0])

import numpy, matplotlib, gymnasium, torch
print("numpy       :", numpy.__version__)
print("matplotlib  :", matplotlib.__version__)
print("gymnasium   :", gymnasium.__version__)
print("torch       :", torch.__version__)
print("CUDA ready  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    major, minor = torch.cuda.get_device_capability(0)
    print("Compute cap : sm_%d%d" % (major, minor))
    print("CUDA version:", torch.version.cuda)
    try:
        x = torch.randn(512, 512, device="cuda")
        _ = (x @ x).sum().item()
        print("GPU matmul  : PASS")
    except Exception as exc:
        print("GPU matmul  : FAIL ->", exc)
else:
    print("NOTE        : GPU not enabled, training will run on CPU (much slower)")

from env_maddpg import MultiAgentEmergencyLaneEnv
env = MultiAgentEmergencyLaneEnv()
obs = env.reset()
print("Env         : agents=%d obs_dim=%d n_actions=%d" % (env.n_agents, env.obs_dim, env.n_actions))
print("Obs shape   :", obs[0].shape)
print()
print("Environment verification PASSED - ready to train.")
'@
Set-Content -LiteralPath $verifyScript -Value $check -Encoding UTF8

Push-Location $ProjectRoot
try {
    & $VenvPython $verifyScript
    $verifyExit = $LASTEXITCODE
} finally {
    Pop-Location
    Remove-Item -LiteralPath $verifyScript -Force -ErrorAction SilentlyContinue
}
if ($verifyExit -ne 0) { throw 'Verification failed, see the errors above.' }

# ---------------------------------------------------------------- done
Write-Step 'Done'
Write-Host @"

  venv      : $VenvDir
  python    : $VenvPython

  NEXT - run these in your terminal:

    # activate the environment (current PowerShell session)
    .\.venv\Scripts\Activate.ps1

    # if script execution is blocked, run this once:
    # Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

    # smoke test: 20 episodes, confirms the training loop works
    python run_training.py --episodes 20 --seed 42

    # full training
    python run_training.py --seed 42

    # watch the log from a second terminal
    Get-Content .\logs_matd3\training_output.log -Wait

  AFTER TRAINING:

    python evaluate.py 50
    python plot_strategy_comparison.py --input evaluation_results.json
    python visualize_training.py

"@ -ForegroundColor Gray
