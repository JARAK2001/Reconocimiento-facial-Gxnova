import face_recognition
import numpy as np
import base64
import io
import os
from PIL import Image, ImageOps
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_to_cloudinary(base64_image: str):
    try:
        print("Iniciando subida a Cloudinary...")
        if ',' in base64_image:
            print("Separando prefijo base64...")
            base64_image = base64_image.split(',')[1]
        
        # Cloudinary uploader can handle base64 directly or file objects
        # We prefix with data:image/png;base64, if not present for the uploader
        upload_result = cloudinary.uploader.upload(
            f"data:image/png;base64,{base64_image}",
            folder="face_recognition"
        )
        url = upload_result.get("secure_url")
        print(f"Subida exitosa: {url}")
        return url
    except Exception as e:
        print(f"Error crítico al subir imagen a Cloudinary: {e}")
        return None

import requests

def url_to_embedding(url: str):
    try:
        print(f"DEBUG: Descargando imagen desde URL: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"ERROR: No se pudo descargar imagen. Status: {response.status_code}")
            return None
        
        image = Image.open(io.BytesIO(response.content))
        image = ImageOps.exif_transpose(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Redimensionar si es muy grande para mejorar velocidad y detección HOG
        max_size = 1024
        if max(image.size) > max_size:
            print(f"DEBUG: Redimensionando imagen de {image.size} a {max_size}px max")
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        image_np = np.array(image)
        
        # Intentar detección con upsampling para caras pequeñas
        face_locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=1, model="hog")
        
        if not face_locations:
            print("WARNING: No se encontró ningún rostro claro.")
            return None
        
        # Generar embedding con alta precisión
        encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations, num_jitters=10)
        
        return encodings[0]
    except Exception as e:
        print(f"CRITICAL ERROR: Error al procesar imagen desde URL: {str(e)}")
        return None

def image_to_embedding(base64_image: str):
    try:
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        image = ImageOps.exif_transpose(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        max_size = 1024
        if max(image.size) > max_size:
            print(f"DEBUG: Redimensionando imagen base64 de {image.size} a {max_size}px max")
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        image_np = np.array(image)
        
        # Intentar detección con upsampling
        face_locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=1, model="hog")
        
        if not face_locations:
            print("WARNING: No se encontró ningún rostro claro en el base64.")
            return None
            
        encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations, num_jitters=10)
        return encodings[0]
    except Exception as e:
        print(f"CRITICAL ERROR: Error al procesar imagen base64: {str(e)}")
        return None
