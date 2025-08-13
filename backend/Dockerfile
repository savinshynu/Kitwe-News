# backend/Dockerfile

# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose backend port
EXPOSE 8000

# Run FastAPI using uvicorn
# set host to 0.0.0.0 so that the api can listen to externally
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]