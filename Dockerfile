# syntax=docker/dockerfile:1
#
# Road Watch scanner image.
#   docker run --rm -v "$PWD:/data" ghcr.io/kprsnt2/road-watch scan ride.mp4 --gpx ride.gpx
#
# Stage 1 exports the pretrained YOLO11n detector to ONNX (needs PyTorch, ~1 GB,
# thrown away). Stage 2 is the small runtime: Python + ONNX Runtime + OpenCV.
# Note: YOLO11 weights are AGPL-3.0 (Ultralytics).

FROM python:3.12-slim AS model
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics onnx onnxslim onnxruntime
WORKDIR /export
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, opset=17, simplify=True)"

FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/kprsnt2/road-watch" \
      org.opencontainers.image.description="Dashcam traffic-violation evidence: detection, plate checks, signed evidence, anonymous hotspots" \
      org.opencontainers.image.licenses="MIT"
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra detect
COPY --from=model /export/yolo11n.onnx /app/models/yolo11n.onnx

ENV PATH=/app/.venv/bin:$PATH ROAD_WATCH_MODEL=/app/models/yolo11n.onnx
WORKDIR /data
ENTRYPOINT ["python", "-m", "road_watch"]
CMD ["--help"]
