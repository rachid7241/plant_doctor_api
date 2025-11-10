"""
FastAPI Application - PlantDoctor Burkina
✅ Configuration optimale, logging, monitoring
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
import sys
from datetime import datetime
from contextlib import asynccontextmanager

# ✅ Configuration du logging AVANT tout le reste
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('plant_doctor.log')
    ]
)

logger = logging.getLogger(__name__)


# ✅ Lifespan events pour startup/shutdown propres
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Démarrage de PlantDoctor API...")
    logger.info(f"⏰ Heure de démarrage: {datetime.now().isoformat()}")
    
    # Charger le service ML
    from app.services.ml_service import ml_service
    status_ml = ml_service.get_model_status()
    logger.info(f"🌱 Service ML: {status_ml['mode']}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt de PlantDoctor API...")


# ✅ Initialisation FastAPI avec configuration optimale
app = FastAPI(
    title="PlantDoctor API Burkina",
    description="API d'analyse des maladies des plantes avec ML",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ✅ Configuration CORS sécurisée
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ✅ Middleware de logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log toutes les requêtes avec timing"""
    start_time = datetime.now()
    
    # Log de la requête
    logger.info(
        f"📥 {request.method} {request.url.path} "
        f"- Client: {request.client.host if request.client else 'unknown'}"
    )
    
    # Traiter la requête
    response = await call_next(request)
    
    # Calculer le temps de traitement
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Log de la réponse
    logger.info(
        f"📤 {request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Temps: {process_time:.3f}s"
    )
    
    # Ajouter header de timing
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# ✅ Gestionnaire d'erreurs de validation
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Gestion propre des erreurs de validation"""
    logger.warning(f"❌ Erreur de validation: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Données invalides",
            "details": exc.errors(),
            "body": exc.body
        }
    )


# ✅ Gestionnaire d'erreurs génériques
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestion globale des erreurs non gérées"""
    logger.exception(f"💥 Erreur non gérée: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Erreur interne du serveur",
            "message": "Une erreur inattendue s'est produite",
            "timestamp": datetime.now().isoformat()
        }
    )


# ✅ Import des routes
from app.routes.analysis import router as analysis_router
from app.routes.weather import router as weather_router

# Inclusion des routes
app.include_router(analysis_router, prefix="/api/v1", tags=["Analysis"])
app.include_router(weather_router, prefix="/api/v1", tags=["Weather"])


# ✅ Routes de base
@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "service": "🌱 PlantDoctor API Burkina",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "analyse_plante": "POST /api/v1/analyze",
            "liste_maladies": "GET /api/v1/diseases",
            "statut_ml": "GET /api/v1/ml-status",
            "meteo": "GET /api/v1/weather",
            "health": "GET /health",
            "documentation": "GET /docs"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    ✅ Health check complet
    Vérifie l'état de tous les services
    """
    from app.services.ml_service import ml_service
    
    ml_status = ml_service.get_model_status()
    
    return {
        "status": "healthy",
        "service": "PlantDoctor API",
        "version": "1.0.0",
        "ml_service": {
            "available": ml_status["service_status"] == "OPERATIONAL",
            "mode": ml_status["mode"]
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    ✅ Métriques basiques de l'application
    """
    from app.services.ml_service import ml_service
    
    ml_status = ml_service.get_model_status()
    
    return {
        "service": "PlantDoctor API",
        "uptime_start": datetime.now().isoformat(),
        "ml_service": ml_status,
        "diseases_count": len(ml_service.MALADIES_BURKINA),
        "timestamp": datetime.now().isoformat()
    }


# ✅ Point d'entrée principal
if __name__ == "__main__":
    logger.info("🚀 Lancement du serveur...")
    
    uvicorn.run(
        "app.main:app",
        host="192.168.56.1",  # ✅ Ton IP
        port=8000,
        reload=True,  # ⚠️ Désactiver en production
        log_level="info",
        access_log=True
    )