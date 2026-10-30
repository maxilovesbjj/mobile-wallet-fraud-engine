FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias primero para aprovechar la caché de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Levantar el servidor
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
