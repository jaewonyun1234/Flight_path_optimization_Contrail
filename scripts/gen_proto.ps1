# Generate the gRPC Python stubs from service/proto/solver.proto into
# service/generated/ (gitignored). PowerShell equivalent of gen_proto.sh,
# for Windows users whose shell has no `bash`.
#
# Usage (from the repo root, with your env active):
#     .\scripts\gen_proto.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "service\generated"
$protoDir = Join-Path $root "service\proto"

New-Item -ItemType Directory -Force -Path $out | Out-Null

python -m grpc_tools.protoc `
    -I $protoDir `
    --python_out=$out `
    --grpc_python_out=$out `
    (Join-Path $protoDir "solver.proto")

# Make the output an importable package whose bare `import solver_pb2` resolves
# (the generated *_pb2_grpc.py imports its sibling without a package prefix).
Set-Content -Path (Join-Path $out "__init__.py") `
    -Value "import os`nimport sys`n`nsys.path.insert(0, os.path.dirname(__file__))`n" `
    -Encoding utf8

Write-Output "Generated gRPC stubs in $out"
