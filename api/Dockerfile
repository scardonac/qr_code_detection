# Usa una imagen base de Python 3.11
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app/api

# Establece el PYTHONPATH para incluir el directorio de trabajo
ENV PYTHONPATH=/app

# Copia los archivos necesarios
COPY ./api /app/api

# Copia la carpeta models desde fuera de api
COPY ./models/best_model /app/api/models/best_model

# Actualiza pip a la última versión
RUN pip install --upgrade pip

# Instala las dependencias del sistema necesarias para OpenCV, zbar y otras librerías
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zbar-tools \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Expone el puerto en el que se ejecutará la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación (usando uvicorn si es FastAPI)
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]



