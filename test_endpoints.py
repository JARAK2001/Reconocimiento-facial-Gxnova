import requests

# URL base del servidor local de FastAPI
BASE_URL = "http://127.0.0.1:8000"

def test_register_face_with_url():
    print("\n--- Probando Registro con URL ---")
    url = f"{BASE_URL}/register-face"
    # Usando una imagen de prueba clara
    data = {
        "nombre": "Test User",
        "imageUrl": "https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg"
    }
    try:
        response = requests.post(url, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_verify_face_with_url():
    print("\n--- Probando Verificación con URL ---")
    url = f"{BASE_URL}/verify-face"
    data = {
        "imageUrl": "https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg"
    }
    try:
        response = requests.post(url, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_register_face_with_json():
    print("\n--- Probando Registro con JSON ---")
    url = f"{BASE_URL}/register-face"
    # Payload en formato JSON
    payload = {
        "nombre": "JSON Test User",
        "imageUrl": "https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg"
    }
    try:
        # requests.post(url, json=payload) envía automáticamente application/json
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Asegúrate de que el servidor FastAPI esté corriendo en http://127.0.0.1:8000")
    # test_register_face_with_url() # Form-data
    test_register_face_with_json() # JSON
    # test_verify_face_with_url() # Form-data
    
    print("\nPrueba completada.")
