import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def test_connection():
    print(f"Cloud Name: {os.getenv('CLOUDINARY_CLOUD_NAME')}")
    print(f"API Key: {os.getenv('CLOUDINARY_API_KEY')}")
    
    try:
        # Intentar subir una imagen pequeña (un pixel transparente)
        # Esto verifica si las credenciales son válidas
        result = cloudinary.uploader.upload(
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
            public_id="test_pixel"
        )
        print("¡Conexión exitosa a Cloudinary!")
        print(f"URL de imagen de prueba: {result.get('secure_url')}")
    except Exception as e:
        print(f"Error al conectar con Cloudinary: {e}")

if __name__ == "__main__":
    test_connection()
