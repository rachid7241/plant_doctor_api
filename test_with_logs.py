import requests
from PIL import Image
import io
import logging

# Configurer les logs pour voir tous les détails
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_with_detailed_logs():
    """
    Test détaillé de l'endpoint /analyze avec logging complet
    """
    print("🧪" * 20)
    print("🧪 TEST DÉTAILLÉ DE L'API ANALYSE")
    print("🧪" * 20)
    
    # URL de ton API
    url = "http://192.168.56.1:8000/api/v1/analyze"
    
    print("📸 Création d'une image test...")
    
    # Créer une image test simple (feuille verte avec taches)
    img = Image.new('RGB', (800, 600), color=(100, 200, 100))  # Fond vert
    
    # Ajouter des "taches" de maladie (points rouges)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Taches de rouille
    draw.ellipse([200, 150, 250, 200], fill=(200, 100, 100), outline='red')  # Tache 1
    draw.ellipse([400, 300, 450, 350], fill=(200, 100, 100), outline='red')  # Tache 2
    draw.ellipse([500, 200, 550, 250], fill=(200, 100, 100), outline='red')  # Tache 3
    
    # Sauvegarder en mémoire
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=85)
    img_bytes.seek(0)  # Retourner au début du fichier
    
    print(f"✅ Image test créée - Taille: {len(img_bytes.getvalue())} bytes")
    
    # CORRECTION : Utiliser 'file' au lieu de 'image'
    files = {'file': ('test_plante_malade.jpg', img_bytes, 'image/jpeg')}
    
    try:
        print("📤 Envoi de la requête POST...")
        print(f"🌐 URL: {url}")
        print(f"📁 Fichier: test_plante_malade.jpg")
        
        # Envoyer la requête avec timeout
        response = requests.post(url, files=files, timeout=30)
        
        print("📥 Réponse reçue!")
        print(f"📊 STATUS CODE: {response.status_code}")
        print(f"📄 HEADERS: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("🎉 SUCCÈS COMPLET!")
            result = response.json()
            
            # CORRECTION : Structure de réponse plus flexible
            print(f"\n📋 RÉSULTATS DE L'ANALYSE:")
            
            # Gérer différentes structures de réponse
            if 'disease' in result:
                if isinstance(result['disease'], dict):
                    print(f"   🌿 Maladie: {result['disease'].get('name', 'N/A')}")
                    print(f"   🎯 Confiance: {result['disease'].get('confidence', 0):.2%}")
                    print(f"   💊 Traitement: {result['disease'].get('treatment', 'N/A')}")
                    print(f"   🛡️  Prévention: {result['disease'].get('prevention', 'N/A')}")
                    print(f"   ⚠️  Urgence: {result['disease'].get('urgency', 'N/A')}")
                else:
                    print(f"   🌿 Maladie: {result['disease']}")
            
            if 'confidence' in result:
                print(f"   🎯 Confiance: {result.get('confidence', 0):.2%}")
            
            if 'weather_impact' in result:
                print(f"   🌤️  Impact météo: {result['weather_impact']}")
            
            if 'recommendation' in result:
                print(f"   💡 Recommandation: {result['recommendation']}")
            
            if 'advice' in result:
                print(f"   💡 Conseil: {result['advice']}")
            
            # Afficher la réponse complète pour debug
            print(f"\n📄 RÉPONSE COMPLÈTE:")
            import json
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        elif response.status_code == 422:
            print("❌ ERREUR 422 - Données non traitable")
            print(f"📝 Détails: {response.text}")
            
        else:
            print(f"❌ ERREUR {response.status_code}")
            print(f"📝 Détails: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("💥 ERREUR: Impossible de se connecter à l'API")
        print("   Vérifie que l'API tourne sur http://192.168.56.1:8000")
        
    except requests.exceptions.Timeout:
        print("⏰ ERREUR: Timeout - L'API met trop de temps à répondre")
        
    except Exception as e:
        print(f"💥 ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()

def test_simple_endpoints():
    """
    Test des autres endpoints pour vérifier que l'API fonctionne
    """
    print("\n" + "🔍" * 20)
    print("🔍 TEST DES AUTRES ENDPOINTS")
    print("🔍" * 20)
    
    base_url = "http://192.168.56.1:8000"
    
    endpoints = [
        "/",
        "/health",
        "/api/v1/ml-status",
        "/api/v1/diseases"
    ]
    
    for endpoint in endpoints:
        url = base_url + endpoint
        try:
            response = requests.get(url, timeout=10)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} {endpoint} - Status: {response.status_code}")
            
            if endpoint == "/api/v1/diseases" and response.status_code == 200:
                data = response.json()
                diseases_count = len(data.get('diseases', []))
                print(f"   📊 {diseases_count} maladies disponibles")
                
            if response.status_code != 200:
                print(f"   Détails: {response.text}")
                
        except Exception as e:
            print(f"❌ {endpoint} - Erreur: {e}")

def test_multiple_formats():
    """
    Test avec différents formats d'image pour vérifier la compatibilité
    """
    print("\n" + "🔄" * 20)
    print("🔄 TEST MULTIFORMATS")
    print("🔄" * 20)
    
    url = "http://192.168.56.1:8000/api/v1/analyze"
    
    # Test avec PNG
    print("📸 Test avec image PNG...")
    img_png = Image.new('RGB', (600, 400), color=(120, 180, 120))
    img_bytes_png = io.BytesIO()
    img_png.save(img_bytes_png, format='PNG')
    img_bytes_png.seek(0)
    
    files_png = {'file': ('test_plante.png', img_bytes_png, 'image/png')}
    
    try:
        response = requests.post(url, files=files_png, timeout=30)
        print(f"📊 PNG - Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Succès: {result.get('disease', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Erreur PNG: {e}")

if __name__ == "__main__":
    # Test des endpoints simples d'abord
    test_simple_endpoints()
    
    # Test de l'analyse d'image
    test_with_detailed_logs()
    
    # Test avec différents formats
    test_multiple_formats()
    
    print("\n" + "🎯" * 20)
    print("🎯 TESTS TERMINÉS")
    print("🎯" * 20)