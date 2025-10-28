# Imagen base estable con soporte de PyTorch
FROM python:3.10-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias para PyTorch + FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto
COPY requirements.txt .
RUN pip install --upgrade pip

# Instalar dependencias de Python
RUN pip install -r requirements.txt

# Copiar el resto del código
COPY . .

# Variable de entorno para Hugging Face Token
ENV HF_TOKEN=""

# Exponer el puerto 
EXPOSE 10000

# Comando de arranque (usar gunicorn para producción)
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:10000", "--timeout", "3600"]
