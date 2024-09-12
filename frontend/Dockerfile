# Use a Python 3.11 base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app/frontend

# Copy the files
COPY . /app/frontend

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "front_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


