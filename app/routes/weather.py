from fastapi import APIRouter, HTTPException
from app.models.schemas import WeatherResponse
import requests
import os
from dotenv import load_dotenv
import logging

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/weather", response_model=WeatherResponse)
async def get_weather(lat: float, lon: float):
    """
    Retourne les données météo pour une localisation
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # Fallback si pas de clé API
    if not api_key:
        logger.info("Utilisation des données météo simulées")
        return get_mock_weather(lat, lon)
    
    try:
        # Appel à l'API OpenWeatherMap
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=fr"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code != 200:
            logger.warning(f"Erreur API météo: {data.get('message', 'Unknown error')}")
            return get_mock_weather(lat, lon)
        
        # Extraction des données
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        conditions = data["weather"][0]["description"]
        location = data.get("name", "Localisation inconnue")
        
        # Génération de recommandations
        recommendation = generate_weather_recommendation(temp, humidity, conditions)
        
        return WeatherResponse(
            temperature=round(temp, 1),
            humidity=humidity,
            conditions=conditions.capitalize(),
            recommendation=recommendation,
            location=location
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération météo: {str(e)}")
        return get_mock_weather(lat, lon)

def get_mock_weather(lat: float, lon: float) -> WeatherResponse:
    """Données météo simulées pour le développement"""
    # Simulation basée sur la localisation (pour varier un peu)
    base_temp = 20 + (lat - 48.0) * 10  # Plus chaud au sud
    temp = round(base_temp + (lon - 2.0), 1)
    humidity = 60 + int(lat * 10) % 40
    
    conditions_options = ["Ensoleillé", "Partiellement nuageux", "Nuageux"]
    conditions = conditions_options[int(lat * 100) % len(conditions_options)]
    
    recommendation = generate_weather_recommendation(temp, humidity, conditions)
    
    return WeatherResponse(
        temperature=temp,
        humidity=humidity,
        conditions=conditions,
        recommendation=recommendation,
        location=f"Position: {lat:.2f}, {lon:.2f}"
    )

def generate_weather_recommendation(temp: float, humidity: int, conditions: str) -> str:
    """Génère une recommandation basée sur les conditions météo"""
    recommendations = []
    
    if "pluie" in conditions.lower() or "pluvieux" in conditions.lower():
        recommendations.append("Évitez les traitements aujourd'hui à cause de la pluie")
    elif temp > 30:
        recommendations.append("Traitement recommandé tôt le matin ou en soirée")
    elif temp < 10:
        recommendations.append("Conditions trop fraîches pour certains traitements")
    else:
        recommendations.append("Bon moment pour traiter les plantes")
    
    if humidity > 80:
        recommendations.append("Conditions humides favorables aux maladies fongiques - surveillez vos plantes")
    elif humidity < 40:
        recommendations.append("Air sec - pensez à arroser régulièrement")
    
    if "vent" in conditions.lower():
        recommendations.append("Évitez les pulvérisations par temps venteux")
    
    return " ".join(recommendations) if recommendations else "Conditions normales pour le jardinage"