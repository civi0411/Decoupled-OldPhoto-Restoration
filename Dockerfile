FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY README.md /app/README.md
COPY configs /app/configs
COPY src /app/src
COPY scripts /app/scripts
COPY app /app/app
COPY docs /app/docs
COPY examples/inputs /app/examples/inputs
COPY examples/golden /app/examples/golden
COPY old_photo_restoration /app/old_photo_restoration
COPY sitecustomize.py /app/sitecustomize.py

# External dependencies and checkpoints are not baked into this image.
# Mount configs/external_paths.yaml, checkpoints/, and any external model folders at runtime.

EXPOSE 7860

CMD ["python", "scripts/run_gradio_demo.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
