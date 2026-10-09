FROM python:3.11-slim
WORKDIR /service
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MODEL_DEVICE=cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY configs/ configs/
COPY scripts/ scripts/
# Trained model weights should be supplied via a mounted volume or downloaded
# from the user's trusted model repos; they are NOT in the source archive.
COPY models/ models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
