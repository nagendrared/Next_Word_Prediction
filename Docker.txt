# Dockerfile - For containerizing the Flask Next Word Prediction app

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for templates
RUN mkdir -p templates
COPY templates/index.html templates/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 5000

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Expose the port the app runs on
EXPOSE $PORT

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app
