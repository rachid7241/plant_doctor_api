import numpy as np
import logging
from PIL import Image
import io
import os

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
        # Classes pour le Burkina Faso
        self.plantes_burkina = ['mil', 'mais', 'coton', 'sorgho']
        self.maladies_burkina = {
            'mil': ['sain', 'rouille', 'charbon', 'cercosporiose'],
            'mais': ['sain', 'mildiou', 'pyrale', 'charbon'],
            'coton': ['sain', 'pourriture', 'bacteriose', 'alternariose'],  # ✅ Corrigé: pas d'accent
            'sorgho': ['sain', 'rouille', 'charbon', 'anthracnose']
        }
        
        self.load_model()
        logger.info("✅ Service ML initialisé")
    
    def load_model(self):
        """Charge le modèle TensorFlow entraîné"""
        model_path = 'ml/models/plant_model_burkina.h5'
        
        # Vérifier si TensorFlow est disponible
        tensorflow_available = self._check_tensorflow()
        
        if not tensorflow_available:
            logger.warning("⚠️ TensorFlow non disponible - Mode simulation")
            return
        
        try:
            if os.path.exists(model_path):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                logger.info("✅ Modèle ML Burkina chargé avec succès")
            else:
                logger.warning(f"📝 Modèle non trouvé: {model_path} - Mode simulation")
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            self.model_loaded = False
    
    def _check_tensorflow(self):
        """Vérifie si TensorFlow est disponible sans planter"""
        try:
            import tensorflow as tf
            logger.info(f"✅ TensorFlow disponible - Version: {tf.__version__}")
            return True
        except ImportError:
            logger.warning("❌ TensorFlow non installé")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Erreur vérification TensorFlow: {e}")
            return False
    
    async def analyze_plant_image(self, image_data: bytes) -> dict:
        """
        Analyse une image de plante et retourne les prédictions
        Utilise le service ML (réel ou simulation)
        """
        try:
            if self.model_loaded and self.model:
                return await self._real_analysis(image_data)
            else:
                return await self._simulated_analysis(image_data)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse ML: {str(e)}")
            # Fallback vers la simulation en cas d'erreur
            return await self._simulated_analysis(image_data)
    
    async def _real_analysis(self, image_data: bytes) -> dict:
        """Analyse avec le vrai modèle TensorFlow"""
        try:
            import tensorflow as tf
            
            # Prétraitement de l'image pour le modèle
            image = Image.open(io.BytesIO(image_data))
            
            # Redimensionner à la taille attendue par le modèle (224x224 standard)
            image = image.resize((224, 224))
            
            # Convertir en array numpy et normaliser
            image_array = np.array(image) / 255.0
            
            # Ajouter dimension batch
            image_array = np.expand_dims(image_array, axis=0)
            
            # Prédiction avec le modèle
            predictions = self.model.predict(image_array)
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index])
            
            # Mapping des prédictions
            maladies = self._get_all_maladies()
            predicted_disease = maladies[predicted_index] if predicted_index < len(maladies) else "Inconnu"
            
            logger.info(f"🔍 Analyse ML réelle - Maladie: {predicted_disease}, Confiance: {confidence:.2f}")
            
            return {
                "predicted_disease": predicted_disease,
                "confidence": confidence,
                "all_predictions": {
                    maladie: float(conf) for maladie, conf in zip(maladies, predictions[0])
                },
                "image_metadata": {
                    "width": image.size[0],
                    "height": image.size[1],
                    "format": image.format,
                    "analysis_type": "REAL_ML"
                },
                "recommendations": self._get_recommendations(predicted_disease)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse ML réelle: {e}")
            raise
    
    async def _simulated_analysis(self, image_data: bytes) -> dict:
        """Analyse simulée en attendant le vrai modèle"""
        try:
            # Chargement de l'image pour obtenir les métadonnées
            image = Image.open(io.BytesIO(image_data))
            image_size = image.size
            
            # Simulation plus réaliste pour le Burkina
            maladies_prioritaires = ["Rouille", "Mildiou", "Charbon", "Cercosporiose", "Plante Sain"]
            
            # Simulation de prédiction avec biais vers les maladies courantes
            base_probs = [0.1, 0.25, 0.2, 0.15, 0.3]  # Probabilités de base
            confidence = np.random.dirichlet(np.array(base_probs) * 10, size=1)[0]
            predicted_index = np.argmax(confidence)
            
            predicted_disease = maladies_prioritaires[predicted_index]
            
            logger.info(f"🔍 Analyse simulée - Maladie: {predicted_disease}, Confiance: {confidence[predicted_index]:.2f}")
            
            return {
                "predicted_disease": predicted_disease,
                "confidence": float(confidence[predicted_index]),
                "all_predictions": {
                    disease: float(conf) for disease, conf in zip(maladies_prioritaires, confidence)
                },
                "image_metadata": {
                    "width": image_size[0],
                    "height": image_size[1],
                    "format": image.format,
                    "analysis_type": "SIMULATION"
                },
                "recommendations": self._get_recommendations(predicted_disease)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse simulée: {e}")
            raise
    
    def _get_all_maladies(self):
        """Retourne toutes les maladies sous forme de liste plate"""
        maladies = []
        for plante, mal_list in self.maladies_burkina.items():
            for maladie in mal_list:
                maladies.append(f"{maladie} ({plante})")
        return maladies
    
    def _get_recommendations(self, disease: str) -> dict:
        """Retourne les recommandations spécifiques pour le Burkina"""
        recommendations = {
            "Rouille": {
                "traitement": "Appliquez un fongicide à base de soufre. Traitement recommandé tôt le matin.",
                "prevention": "Évitez les densités de plantation trop élevées. Pratiquez la rotation des cultures.",
                "urgence": "medium"
            },
            "Mildiou": {
                "traitement": "Utilisez un fongicide systémique. Évitez les arrosages par aspersion.",
                "prevention": "Assurez une bonne circulation d'air. Utilisez des variétés résistantes.",
                "urgence": "high"
            },
            "Charbon": {
                "traitement": "Traitement fongicide préventif. Brûlez les plants atteints.",
                "prevention": "Utilisez des semences saines. Pratiquez la rotation sur 3 ans.",
                "urgence": "high"
            },
            "Cercosporiose": {
                "traitement": "Fongicides à base de triazoles. Répétez le traitement après pluie.",
                "prevention": "Évitez l'humidité prolongée sur les feuilles.",
                "urgence": "medium"
            },
            "Plante Sain": {
                "traitement": "Aucun traitement nécessaire. Continuez les bonnes pratiques.",
                "prevention": "Maintenez la surveillance régulière. Fertilisation équilibrée.",
                "urgence": "low"
            }
        }
        
        # Chercher la recommandation par mot-clé
        for key in recommendations:
            if key.lower() in disease.lower():
                return recommendations[key]
        
        # Fallback
        return {
            "traitement": "Surveillance recommandée. Consultez un agronome local.",
            "prevention": "Pratiques culturales adaptées au climat burkinabè.",
            "urgence": "low"
        }
    
    def get_model_status(self) -> dict:
        """Retourne le statut du service ML"""
        tensorflow_available = self._check_tensorflow()
        
        return {
            "model_loaded": self.model_loaded,
            "tensorflow_available": tensorflow_available,
            "mode": "REAL_ML" if self.model_loaded else "SIMULATION",
            "classes_burkina": self.maladies_burkina,
            "service_status": "OPERATIONAL"
        }

# Instance globale du service ML
ml_service = MLService()