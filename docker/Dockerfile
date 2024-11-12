FROM python:3.10

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia todos los archivos del directorio actual al contenedor en /app
COPY . /app

# Instala las dependencias de requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar el script por defecto
CMD ["python", "cuentas.py"]