# Usa una imagen base de Python 3.11
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app/frontend

# Copia los archivos necesarios
COPY . /app/frontend

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto que utilizará Streamlit (por defecto es el 8501)
EXPOSE 8501

# Comando para ejecutar la aplicación con Streamlit
CMD ["streamlit", "run", "front_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


