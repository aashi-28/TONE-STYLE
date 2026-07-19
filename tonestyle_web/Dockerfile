# Dockerfile for ToneStyle Django app — Hugging Face Spaces deployment

FROM python:3.11-slim

# System dependencies needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Collect static files at build time
RUN python manage.py collectstatic --noinput

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

# Limit TF/OpenMP threads to reduce memory pressure, same as before
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1

CMD ["gunicorn", "tonestyle_web.wsgi", "--bind", "0.0.0.0:7860", "--timeout", "120"]