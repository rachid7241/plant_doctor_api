from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import AnalysisResponse, DiseaseResponse
from app.services.ml_service import ml_service  # ✅ Import du service ML
import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_plant(file: UploadFile = File(...)):
    """
    Analyse une image de plante et retourne le diagnostic
    Utilise le service ML (réel ou simulation)
    """
    # Validation du fichier
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Le fichier doit être une image (JPEG, PNG)"
        )
    
    logger.info(f"🔍 Analyse d'image reçue: {file.filename}")
    
    try:
        # Lire les données de l'image
        image_data = await file.read()
        
        # Analyser avec le service ML (réel ou simulation)
        ml_result = await ml_service.analyze_plant_image(image_data)
        
        logger.info(f"✅ Analyse terminée - Maladie: {ml_result['predicted_disease']}, Confiance: {ml_result['confidence']:.2f}")
        
        # Génération de recommandations contextuelles
        weather_impact, recommendation = generate_recommendations(ml_result["predicted_disease"])
        
        return AnalysisResponse(
            disease=DiseaseResponse(
                name=ml_result["predicted_disease"],
                confidence=ml_result["confidence"],
                treatment=ml_result["recommendations"]["traitement"],
                prevention=ml_result["recommendations"]["prevention"],
                urgency=ml_result["recommendations"]["urgence"]
            ),
            weather_impact=weather_impact,
            recommendation=recommendation,
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'analyse")

@router.get("/diseases")
async def get_diseases_list():
    """
    Retourne la liste de toutes les maladies connues
    """
    # Utilise maintenant les maladies du service ML
    maladies_burkina = ml_service.maladies_burkina
    
    # Transformer la structure pour l'API
    diseases_list = []
    for plante, maladies in maladies_burkina.items():
        for maladie in maladies:
            diseases_list.append({
                "key": f"{plante}_{maladie}",
                "name": f"{maladie.capitalize()} ({plante.capitalize()})",
                "plante": plante,
                "maladie": maladie
            })
    
    return {
        "count": len(diseases_list),
        "diseases": diseases_list,
        "mode_analyse": "ML" if ml_service.model_loaded else "SIMULATION"
    }

@router.get("/ml-status")
async def get_ml_status():
    """
    Retourne le statut du service Machine Learning
    """
    status = ml_service.get_model_status()
    
    return {
        "service_ml": "🌱 PlantDoctor Burkina ML",
        "statut": status,
        "timestamp": datetime.datetime.now().isoformat()
    }

def generate_recommendations(disease_name: str) -> tuple[str, str]:
    """Génère des recommandations basées sur la maladie et les conditions"""
    
    # Recommandations météo basées sur la maladie
    weather_recommendations = {
        "Rouille": (
            "Conditions humides favorables au développement de la rouille",
            "Traitement recommandé tôt le matin par temps sec et stable"
        ),
        "Mildiou": (
            "Températures fraîches et humidité élevée - conditions idéales pour le mildiou",
            "Appliquez le traitement en fin de journée, évitez les périodes de pluie"
        ),
        "Charbon": (
            "Conditions chaudes et humides favorables au charbon",
            "Traitement préventif recommandé avant les périodes pluvieuses"
        ),
        "Cercosporiose": (
            "Humidité persistante favorable à la cercosporiose", 
            "Traitement efficace par temps sec après la rosée du matin"
        ),
        "Pucerons": (
            "Conditions printanières favorables aux pucerons",
            "Traitement efficace par temps calme et sec, tôt le matin"
        ),
        "Plante Sain": (
            "Conditions optimales pour la croissance",
            "Continuez les bonnes pratiques, surveillance régulière recommandée"
        )
    }
    
    # Trouver la recommandation la plus proche
    for key, value in weather_recommendations.items():
        if key.lower() in disease_name.lower():
            return value
    
    # Fallback pour maladies inconnues
    return (
        "Conditions de croissance normales",
        "Surveillance et pratiques culturales adaptées recommandées"
    )

# Fonction pour sauvegarder en base de données (optionnelle)
def save_analysis_to_db(filename: str, disease_name: str, confidence: float, treatment: str):
    """
    Sauvegarde l'analyse dans la base de données
    (À décommenter quand ta base de données sera configurée)
    """
    try:
        # Décommente ces lignes quand tu auras configuré ta base
        # from app.database.database import SessionLocal, AnalysisHistory
        # 
        # db = SessionLocal()
        # try:
        #     analysis_record = AnalysisHistory(
        #         image_filename=filename,
        #         disease_name=disease_name,
        #         confidence=confidence,
        #         treatment=treatment,
        #         location="Burkina Faso"
        #     )
        #     db.add(analysis_record)
        #     db.commit()
        #     logger.info(f"✅ Analyse sauvegardée en base: {filename}")
        # finally:
        #     db.close()
        pass
        
    except Exception as e:
        logger.warning(f"⚠️ Impossible de sauvegarder en base: {e}")