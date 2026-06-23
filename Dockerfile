# syntax=docker/dockerfile:1
#
# Containerized CP-SAT solver service (gRPC Solve RPC + ZMQ progress stream).
#
# The image ships ONLY the headless solver runtime — numpy / OR-Tools / gRPC /
# ZMQ — and NOT the PyQt6 desktop dashboard, so it stays small and needs no
# GUI/X11 system libraries. Build once, run anywhere the ports are reachable:
#
#   docker build -t contrail-solver .
#   docker run --rm -p 50051:50051 -p 5556:5556 contrail-solver
#
# ---------------------------------------------------------------------------
# Stage 1: builder — install dependencies into an isolated virtualenv and
# generate the gRPC stubs from the .proto. Kept separate so the runtime image
# carries only the finished venv + source, never pip's build caches.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# All third-party deps ship manylinux wheels for CPython 3.11, so no compiler
# toolchain is needed — the slim base is enough.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependency layer first: these inputs change rarely, so Docker reuses the
# cached `pip install` whenever only application code changes.
COPY pyproject.toml README.md ./
COPY contrail_env/ ./contrail_env/
COPY service/ ./service/
COPY scripts/ ./scripts/
# Only contrail_env + service ship here — the image's one job is the CP-SAT gRPC
# server. The quantum solver modules (pasqal_analog, xanadu_gbs) ride along in
# contrail_env and run on their built-in fallbacks; the optional [quantum] SDKs
# (pulser/strawberryfields) are not installed, keeping the runtime image lean.
RUN pip install .

# The gRPC stubs are gitignored — generate them from solver.proto at build time.
RUN bash scripts/gen_proto.sh

# ---------------------------------------------------------------------------
# Stage 2: runtime — copy the ready-built venv + source, run as a non-root
# user, and expose the service ports.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# PYTHONPATH=/app makes the source tree (which carries the freshly generated
# service/generated stubs) take precedence over the installed copy.
# CONTRAIL_GRPC_HOST=0.0.0.0 makes the bound port reachable from outside the
# container; local runs still default to localhost (see service/server.py).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    CONTRAIL_GRPC_HOST=0.0.0.0 \
    CONTRAIL_GRPC_PORT=50051

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Least privilege: drop root and run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 50051 = gRPC Solve RPC, 5556 = ZMQ progress publisher.
EXPOSE 50051 5556

# Liveness probe: the gRPC port is accepting TCP connections. (A full gRPC
# health service would need grpc_health_probe; a socket connect is enough to
# tell the orchestrator the process is up and listening.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 50051), 2).close()" || exit 1

CMD ["python", "-m", "service.server"]
