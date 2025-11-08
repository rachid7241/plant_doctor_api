from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialisation de l'application
app = FastAPI(
    title="PlantDoctor API",
    description="API d'analyse des maladies des plantes avec recommandations",
    version="1.0.0"
)

# Configuration CORS pour l'app Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À changer en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import DIRECT des routes
from app.routes.analysis import router as analysis_router
from app.routes.weather import router as weather_router

# Inclusion des routes
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])
app.include_router(weather_router, prefix="/api/v1", tags=["weather"])

@app.get("/")
async def root():
    return {
        "message": "🌱 PlantDoctor API est opérationnelle!",
        "endpoints": {
            "analyse_plante": "POST /api/v1/analyze",
            "meteo": "GET /api/v1/weather",
            "documentation": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PlantDoctor API"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="192.168.56.1",
        port=8000,
        reload=True
    )