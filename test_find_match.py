import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_find_match():
    print(f"\n--- Probando /find-match ---")
    url = f"{BASE_URL}/find-match"
    
    # 1. Target URL (foto de la persona intentando entrar)
    target_url = "https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg" # Ojo: esto no es una cara, face_recognition va a fallar
    # Cambiemos a una imagen pública con cara
    target_url = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"
    
    # 2. Candidate URLs (fotos que ya están en la DB)
    candidate_urls = [
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg" # Match
    ]
    
    data = {
        "targetUrl": target_url,
        "candidateUrls": candidate_urls
    }
    
    # Intento 1 (Cálculo inicial, debe demorar)
    print("Intento 1: Sin caché...")
    start_time = time.time()
    try:
        response = requests.post(url, json=data)
        elapsed = time.time() - start_time
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"Tiempo total Intento 1: {elapsed:.2f} segundos")
    except Exception as e:
        print(f"Error: {e}")

    # Intento 2 (Con caché, debe ser instantáneo)
    print("\nIntento 2: Con caché activa...")
    start_time = time.time()
    try:
        response = requests.post(url, json=data)
        elapsed = time.time() - start_time
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"Tiempo total Intento 2: {elapsed:.2f} segundos")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_find_match()
