# ============================================================================
# FICHIER CORRIGÉ: config.py - Configuration finale
# ============================================================================

import os

# Configuration globale du système multi-agent agricole

# Adresses des agents
AGENT_ADDRESSES = {
    "coordinator": "coordinator@localhost",
    "data_manager": "soilmanager@localhost", 
    "logistic": "logistic@localhost",
    "svm": "svm@localhost",
    "random_forest": "forest@localhost",
    "knn": "knn@localhost",
    "neural_network": "neural@localhost",
    "comparator": "comparator@localhost",
    "visualizer": "visualizer@localhost"
}

# Paramètres des données - CORRIGÉ selon votre fichier
DATA_CONFIG = {
    "file_path": "datafinal1.csv",  # Votre fichier dans le répertoire principal
    "test_size": 0.2,
    "random_state": 42,
    "features": [
        "Nitrogen", "phosphorous", "Potassium", 
        "temperature", "humidity", "ph",
        "Rainfall Mensuel (mm)", "Rainfall Annuel (mm)"
    ],
    "target": "besoin_irrigation"
}

# Paramètres des modèles ML - optimisés
MODEL_PARAMS = {
    "logistic": {
        "max_iter": 1000,
        "random_state": 42,
        "solver": "lbfgs"
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "random_state": 42,
        "gamma": "scale"
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": 10,
        "min_samples_split": 5
    },
    "knn": {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto"
    },
    "neural_network": {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 300,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1
    }
}

# Chemins de sauvegarde
PATHS = {
    "data": "./",
    "models": "results/models/",
    "metrics": "results/metrics/",
    "plots": "results/plots/",
    "reports": "results/reports/"
}

# Configuration SPADE
SPADE_CONFIG = {
    "password": "password",  # Mot de passe uniforme
    "timeout": 60
}

# Fonction utilitaire
def ensure_directories():
    """Créer tous les dossiers nécessaires"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    os.makedirs("agents", exist_ok=True)
    os.makedirs("utils", exist_ok=True)
        

        