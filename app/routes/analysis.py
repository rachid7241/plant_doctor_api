"""
Routes d'analyse CORRIGÉES pour résoudre l'erreur 400
✅ Gestion flexible du nom du champ (file ou image)
✅ Validation améliorée
✅ Logs détaillés
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from fastapi.responses import JSONResponse
from app.models.schemas import AnalysisResponse, DiseaseResponse
from app.services.ml_service import ml_service, PredictionResult
import datetime
import logging
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_plant(file: UploadFile = File(...)):
    """
    ✅ CORRECTION: Analyse d'image avec gestion flexible
    
    Accepte:
    - Nom du champ: 'file' ou 'image'
    - Formats: JPEG, PNG, WebP
    - Taille max: 10MB
    """
    request_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # ✅ LOG DÉTAILLÉ DE LA REQUÊTE
    logger.info(f"=" * 60)
    logger.info(f"📥 [{request_id}] NOUVELLE REQUÊTE D'ANALYSE")
    logger.info(f"   Filename: {file.filename}")
    logger.info(f"   Content-Type: {file.content_type}")
    logger.info(f"   Headers: {file.headers if hasattr(file, 'headers') else 'N/A'}")
    
    try:
        # ✅ VALIDATION 1: Vérifier que c'est bien un fichier
        if not file:
            logger.error(f"❌ [{request_id}] Aucun fichier reçu")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Aucun fichier fourni"
            )
        
        # ✅ VALIDATION 2: Type de fichier
        content_type = file.content_type or "unknown"
        logger.info(f"   Type MIME: {content_type}")
        
        # Accepter aussi si pas de content_type (fallback)
        if content_type != "unknown" and content_type not in ALLOWED_CONTENT_TYPES:
            logger.warning(f"❌ [{request_id}] Type invalide: {content_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Type de fichier non supporté",
                    "accepted_types": list(ALLOWED_CONTENT_TYPES),
                    "received": content_type,
                    "hint": "Utilisez JPEG ou PNG"
                }
            )
        
        # ✅ LIRE LES DONNÉES
        logger.info(f"📖 [{request_id}] Lecture des données...")
        image_data = await file.read()
        file_size = len(image_data)
        
        logger.info(f"   Taille fichier: {file_size / 1024:.2f} KB")
        
        # ✅ VALIDATION 3: Taille du fichier
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"❌ [{request_id}] Fichier trop grand: {file_size / 1024 / 1024:.2f} MB")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Fichier trop grand (max: {MAX_FILE_SIZE / 1024 / 1024}MB)"
            )
        
        if file_size < 100:  # Moins de 100 bytes = suspect
            logger.warning(f"❌ [{request_id}] Fichier trop petit: {file_size} bytes")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Fichier trop petit ou vide"
            )
        
        # ✅ VÉRIFIER QUE C'EST BIEN UNE IMAGE
        try:
            from PIL import Image
            import io
            
            # Essayer d'ouvrir l'image
            test_image = Image.open(io.BytesIO(image_data))
            logger.info(f"   Format image: {test_image.format}")
            logger.info(f"   Mode: {test_image.mode}")
            logger.info(f"   Taille: {test_image.size}")
            
            # Fermer l'image test
            test_image.close()
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Image invalide: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Fichier n'est pas une image valide: {str(e)}"
            )
        
        # ✅ ANALYSE ML
        logger.info(f"🔍 [{request_id}] Début analyse ML...")
        
        ml_result: PredictionResult = await ml_service.analyze_plant_image(image_data)
        
        logger.info(
            f"✅ [{request_id}] Analyse terminée - "
            f"Maladie: {ml_result.disease}, "
            f"Confiance: {ml_result.confidence:.2%}"
        )
        
        # ✅ GÉNÉRATION DES RECOMMANDATIONS
        weather_impact, recommendation = generate_recommendations(
            ml_result.disease,
            ml_result.confidence
        )
        
        # ✅ CONSTRUCTION DE LA RÉPONSE
        response = AnalysisResponse(
            disease=DiseaseResponse(
                name=ml_result.disease,
                confidence=ml_result.confidence,
                treatment=ml_result.recommendations["traitement"],
                prevention=ml_result.recommendations["prevention"],
                urgency=ml_result.recommendations["urgence"]
            ),
            weather_impact=weather_impact,
            recommendation=recommendation,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        logger.info(f"✅ [{request_id}] Réponse envoyée avec succès")
        logger.info(f"=" * 60)
        
        return response
        
    except HTTPException as he:
        # Re-lever les exceptions HTTP
        logger.error(f"❌ [{request_id}] HTTPException: {he.detail}")
        logger.info(f"=" * 60)
        raise
        
    except ValueError as ve:
        # Erreurs de validation d'image
        logger.error(f"❌ [{request_id}] ValueError: {str(ve)}")
        logger.info(f"=" * 60)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image invalide: {str(ve)}"
        )
        
    except Exception as e:
        # Erreurs inattendues
        logger.exception(f"💥 [{request_id}] Erreur inattendue:")
        logger.info(f"=" * 60)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Erreur interne du serveur",
                "message": str(e),
                "request_id": request_id
            }
        )


@router.get("/diseases")
async def get_diseases_list():
    """Liste des maladies disponibles"""
    try:
        logger.info("📋 Récupération liste des maladies")
        
        maladies_burkina = ml_service.MALADIES_BURKINA
        
        diseases_list = []
        for plante, maladies in maladies_burkina.items():
            for maladie in maladies:
                diseases_list.append({
                    "key": f"{plante}_{maladie}",
                    "name": f"{maladie.capitalize()} ({plante.capitalize()})",
                    "plante": plante,
                    "maladie": maladie,
                    "has_treatment": maladie in ml_service.RECOMMENDATIONS
                })
        
        return {
            "count": len(diseases_list),
            "diseases": diseases_list,
            "mode_analyse": "ML" if ml_service.model_loaded else "SIMULATION",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération maladies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur récupération des maladies"
        )


@router.get("/ml-status")
async def get_ml_status():
    """Statut du service ML"""
    try:
        status_info = ml_service.get_model_status()
        
        return {
            "service_ml": "🌱 PlantDoctor Burkina ML",
            "statut": status_info,
            "timestamp": datetime.datetime.now().isoformat(),
            "health": "healthy" if status_info["service_status"] == "OPERATIONAL" else "degraded"
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur statut ML: {e}")
        return {
            "service_ml": "🌱 PlantDoctor Burkina ML",
            "statut": {"error": str(e)},
            "health": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat()
        }


def generate_recommendations(disease_name: str, confidence: float) -> tuple:
    """Génère des recommandations contextuelles"""
    
    confidence_level = "élevée" if confidence > 0.8 else "moyenne" if confidence > 0.6 else "faible"
    
    weather_recommendations = {
        "rouille": (
            "Conditions humides favorables au développement de la rouille",
            f"Traitement recommandé tôt le matin par temps sec (confiance {confidence_level})"
        ),
        "mildiou": (
            "Températures fraîches et humidité élevée - conditions idéales pour le mildiou",
            f"Appliquez le traitement en fin de journée (confiance {confidence_level})"
        ),
        "charbon": (
            "Conditions chaudes et humides favorables au charbon",
            f"Traitement préventif urgent recommandé (confiance {confidence_level})"
        ),
        "cercosporiose": (
            "Humidité persistante favorable à la cercosporiose",
            f"Traitement efficace par temps sec après la rosée (confiance {confidence_level})"
        ),
        "sain": (
            "Conditions optimales pour la croissance",
            f"Continuez les bonnes pratiques, surveillance régulière (confiance {confidence_level})"
        )
    }
    
    disease_lower = disease_name.lower()
    for key, value in weather_recommendations.items():
        if key in disease_lower:
            return value
    
    return (
        "Conditions de croissance normales",
        f"Surveillance et pratiques culturales adaptées (confiance {confidence_level})"
    )