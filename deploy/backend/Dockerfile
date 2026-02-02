FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install system dependencies for image processing
# libgl1 and libglib2.0-0 required for OpenCV/Pillow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Expose port 7860 (HuggingFace Spaces standard)
EXPOSE 7860

# Run Gunicorn WSGI server with 4 workers
# Increased timeout (120s) for model loading and image processing
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "--timeout", "120", "app:helmet_api"]
