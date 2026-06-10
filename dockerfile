FROM ultralytics/ultralytics:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY . .

RUN uv pip install --system --no-cache-dir --no-deps ultralytics -e .

ENTRYPOINT ["spark-detector"]
