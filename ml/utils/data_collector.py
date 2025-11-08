import os
import requests
from PIL import Image
import io

class DataCollector:
    def __init__(self, base_path="ml/datasets/burkina"):
        self.base_path = base_path
        self.create_folders()
    
    def create_folders(self):
        """Crée la structure de dossiers pour le Burkina"""
        categories = {
            'mil': ['sain', 'rouille', 'charbon'],
            'mais': ['sain', 'mildiou', 'pyrale'],
            'coton': ['sain', 'pourriture', 'bacterose']
        }
        
        for plante, maladies in categories.items():
            for maladie in maladies:
                path = os.path.join(self.base_path, 'train', plante, maladie)
                os.makedirs(path, exist_ok=True)
                
                path = os.path.join(self.base_path, 'test', plante, maladie)
                os.makedirs(path, exist_ok=True)
        
        print("✅ Structure de dossiers créée pour le Burkina Faso")
    
    def count_images(self):
        """Compte le nombre d'images par catégorie"""
        counts = {}
        for split in ['train', 'test']:
            split_path = os.path.join(self.base_path, split)
            counts[split] = {}
            
            for plante in os.listdir(split_path):
                plante_path = os.path.join(split_path, plante)
                if os.path.isdir(plante_path):
                    counts[split][plante] = {}
                    
                    for maladie in os.listdir(plante_path):
                        maladie_path = os.path.join(plante_path, maladie)
                        if os.path.isdir(maladie_path):
                            num_images = len([f for f in os.listdir(maladie_path) 
                                            if f.endswith(('.jpg', '.jpeg', '.png'))])
                            counts[split][plante][maladie] = num_images
        
        return counts

# Test du collecteur
if __name__ == "__main__":
    collector = DataCollector()
    print(collector.count_images())