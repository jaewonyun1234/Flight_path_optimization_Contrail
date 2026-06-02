#!/usr/bin/env bash
# Generate the gRPC Python stubs from service/proto/solver.proto into
# service/generated/ (which is gitignored — always regenerate from the .proto).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/service/generated"
PROTO_DIR="$ROOT/service/proto"

mkdir -p "$OUT"

python -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    --python_out="$OUT" \
    --grpc_python_out="$OUT" \
    "$PROTO_DIR/solver.proto"

# Make the output an importable package. The generated *_pb2_grpc.py does a
# bare `import solver_pb2`, so the shim puts its own directory on sys.path,
# letting `from service.generated import solver_pb2_grpc` resolve cleanly.
cat > "$OUT/__init__.py" <<'PY'
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
PY

echo "Generated gRPC stubs in $OUT"
