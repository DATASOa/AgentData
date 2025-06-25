
# ============================================================================
# FICHIER 2: config.py - Configuration
# ============================================================================

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

# Paramètres des données
DATA_CONFIG = {
    "file_path": "data/datafinal1.csv",
    "test_size": 0.2,
    "random_state": 42,
    "features": [
        "Nitrogen", "phosphorous", "Potassium", 
        "temperature", "humidity", "ph",
        "Rainfall Mensuel (mm)", "Rainfall Annuel (mm)"
    ],
    "target": "besoin_irrigation"
}

# Paramètres des modèles ML
MODEL_PARAMS = {
    "logistic": {
        "max_iter": 1000,
        "random_state": 42
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": 10
    },
    "knn": {
        "n_neighbors": 5,
        "weights": "uniform"
    },
    "neural_network": {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 200,
        "random_state": 42
    }
}

# Chemins de sauvegarde
PATHS = {
    "data": "data/",
    "models": "results/models/",
    "metrics": "results/metrics/",
    "plots": "results/plots/",
    "reports": "results/reports/"
}

# Types de cultures
CROP_TYPES = {
    "cereals": ["rice", "maize"],
    "legumes": ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"],
    "fruits": ["banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya", "coconut"],
    "others": ["pomegranate", "cotton", "jute", "coffee"]
} 