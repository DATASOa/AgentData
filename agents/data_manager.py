# ============================================================================
# FICHIER 6: agents/data_manager_agent.py - Gestion des données
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
import json
import joblib
import os
from config import DATA_CONFIG, AGENT_ADDRESSES

class SoilDataManagerAgent(Agent):
    """📊 Agent gestionnaire des données de sol agricole"""
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.scaler = StandardScaler()
        self.data_processed = False
    
    class ProcessSoilDataBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            
            if msg and msg.get_metadata("ontology") == "start_processing":
                print("\n📊 [SOIL DATA] Début du traitement des données agricoles")
                
                try:
                    # Charger les données
                    print("📁 [SOIL DATA] Chargement du fichier CSV...")
                    data = pd.read_csv(DATA_CONFIG["file_path"])
                    print(f"✅ [SOIL DATA] {len(data)} échantillons de sols chargés")
                    
                    # Analyser les données
                    print("🔍 [SOIL DATA] Analyse des données...")
                    features = DATA_CONFIG["features"]
                    target = DATA_CONFIG["target"]
                    
                    print(f"📊 [SOIL DATA] Features: {len(features)} paramètres de sol")
                    print(f"🎯 [SOIL DATA] Target: {target}")
                    
                    # Préparer les features et target
                    X = data[features]
                    y = data[target]
                    
                    # Vérifications
                    print(f"🧪 [SOIL DATA] Valeurs manquantes: {X.isnull().sum().sum()}")
                    print(f"💧 [SOIL DATA] Irrigation nécessaire: {y.sum()}/{len(y)} cultures")
                    
                    # Normalisation
                    print("⚖️ [SOIL DATA] Normalisation des paramètres de sol...")
                    X_scaled = self.agent.scaler.fit_transform(X)
                    
                    # Division train/test
                    print("✂️ [SOIL DATA] Division train/test...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, 
                        test_size=DATA_CONFIG["test_size"], 
                        random_state=DATA_CONFIG["random_state"],
                        stratify=y
                    )
                    
                    print(f"📈 [SOIL DATA] Entraînement: {len(X_train)} échantillons")
                    print(f"📊 [SOIL DATA] Test: {len(X_test)} échantillons")
                    
                    # Sauvegarder le scaler
                    os.makedirs("results/models", exist_ok=True)
                    joblib.dump(self.agent.scaler, "results/models/soil_scaler.pkl")
                    
                    # Préparer les données pour envoi
                    soil_data = {
                        "X_train": X_train.tolist(),
                        "X_test": X_test.tolist(),
                        "y_train": y_train.tolist(),
                        "y_test": y_test.tolist(),
                        "feature_names": features,
                        "target_name": target,
                        "train_size": len(X_train),
                        "test_size": len(X_test),
                        "irrigation_rate": float(y.mean()),
                        "crops": data["label"].unique().tolist()
                    }
                    
                    # Envoyer aux 5 agents de modèles
                    model_agents = ["logistic@localhost", "svm@localhost", "forest@localhost", 
                                  "knn@localhost", "neural@localhost"]
                    
                    for agent_address in model_agents:
                        msg = Message(to=agent_address)
                        msg.set_metadata("performative", "inform")
                        msg.set_metadata("ontology", "soil_data")
                        msg.body = json.dumps(soil_data)
                        await self.send(msg)
                        
                        agent_name = agent_address.split("@")[0]
                        print(f"📤 [SOIL DATA] Données envoyées à {agent_name}")
                    
                    print("✅ [SOIL DATA] Toutes les données distribuées aux modèles")
                    self.agent.data_processed = True
                    
                except Exception as e:
                    print(f"❌ [SOIL DATA] Erreur: {e}")
    
    async def setup(self):
        print("📊 [SOIL DATA] Agent gestionnaire de données agricoles initialisé")
        template = Template()
        template.set_metadata("ontology", "start_processing")
        self.add_behaviour(self.ProcessSoilDataBehaviour(), template)  
         