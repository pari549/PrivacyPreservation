FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    MPLCONFIGDIR=/workspace/.cache/matplotlib

WORKDIR /workspace/privacy_preservation

# OpenCV's Python wheel needs these runtime libraries on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./

# The default environment mirrors the current CPU-only workstation. Rebuild
# with TORCH_INDEX_URL set to a supported CUDA wheel index for NVIDIA hosts.
RUN python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple "torch==2.14.0" \
    && python -m pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspace/privacy_preservation

CMD ["python", "-c", "import time; time.sleep(10**8)"]
