FROM python:3.10-slim

WORKDIR /app

# Install deps first so Docker caches this layer between rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + serialized model.
COPY app.py .
COPY superkart_model.joblib .

EXPOSE 7860
CMD ["python", "app.py"]
