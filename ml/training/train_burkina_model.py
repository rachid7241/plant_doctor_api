import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt

class BurkinaPlantModel:
    def __init__(self, num_classes=8):  # 7 maladies + sain
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def create_model(self):
        """Crée un modèle optimisé pour les plantes du Burkina"""
        # Utiliser MobileNetV2 pré-entraîné (léger et efficace)
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Geler les couches de base
        base_model.trainable = False
        
        # Ajouter nos couches de classification
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compiler le modèle
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modèle créé pour les plantes du Burkina")
        return self.model
    
    def prepare_data(self, data_dir, batch_size=32):
        """Prépare les données d'entraînement"""
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20% pour la validation
        )
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        print(f"✅ Données préparées - {train_generator.samples} images d'entraînement")
        print(f"✅ {validation_generator.samples} images de validation")
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=20):
        """Entraîne le modèle"""
        print("🚀 Début de l'entraînement...")
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
            ]
        )
        
        print("✅ Entraînement terminé!")
        return self.history
    
    def save_model(self, model_path='ml/models/plant_model_burkina.h5'):
        """Sauvegarde le modèle entraîné"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"✅ Modèle sauvegardé: {model_path}")
    
    def plot_training(self):
        """Affiche les courbes d'apprentissage"""
        if self.history is None:
            print("❌ Aucun historique d'entraînement")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Courbe de précision
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Courbe de loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('ml/training/training_history.png')
        plt.show()

# Entraînement du modèle
if __name__ == "__main__":
    # Initialiser le modèle
    burkina_model = BurkinaPlantModel(num_classes=8)
    
    # Créer l'architecture
    model = burkina_model.create_model()
    
    # Préparer les données
    train_gen, val_gen = burkina_model.prepare_data('ml/datasets/burkina')
    
    # Afficher le résumé du modèle
    model.summary()
    
    # Entraîner (décommente quand tu as des données)
    # history = burkina_model.train(train_gen, val_gen, epochs=10)
    # burkina_model.save_model()
    # burkina_model.plot_training()