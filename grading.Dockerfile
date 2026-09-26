FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl graphviz \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (~200MB vs ~2GB with CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY grading_service/requirements.txt /app/requirements.txt
# The Aliyun host reaches pypi.org at ~10 KB/s and times out; use the Aliyun mirror instead.
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

COPY torch_judge/ /app/torch_judge/
COPY grading_service/ /app/grading_service/
COPY pyproject.toml /app/pyproject.toml

RUN mkdir -p /app/data

ENV DB_PATH=/app/data/pyre.db

EXPOSE 8000

CMD ["uvicorn", "grading_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
