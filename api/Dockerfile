# Use a Python 3.11 base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app/api

# Set the PYTHONPATH to include the working directory
ENV PYTHONPATH=/app

# Copy the files
COPY ./api /app/api

# Copy the models folder containing the trained model
COPY ./models/best_model /app/api/models/best_model

# Update pip to the latest version
RUN pip install --upgrade pip

# Install the necessary system dependencies for OpenCV, zbar and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zbar-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]



