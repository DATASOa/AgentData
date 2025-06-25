# ============================================================================
# FICHIER 10: agents/knn_agent.py - KNN
# ============================================================================

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
import json
import time
import joblib
from config import MODEL_PARAMS

class KNNSoilAgent(Agent):
    """📍 Agent KNN pour similarité des profils de sol"""
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.model = None
        self.results = None
    
    class TrainKNNBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            
            if msg and msg.get_metadata("ontology") == "soil_data":
                print("\n📍 [KNN] Analyse par similarité des sols")
                
                try:
                    # Décoder les données
                    soil_data = json.loads(msg.body)
                    X_train = np.array(soil_data["X_train"])
                    X_test = np.array(soil_data["X_test"])
                    y_train = np.array(soil_data["y_train"])
                    y_test = np.array(soil_data["y_test"])
                    
                    k = MODEL_PARAMS["knn"]["n_neighbors"]
                    print(f"🎯 [KNN] Recherche des {k} sols les plus similaires")
                    print(f"📊 [KNN] Base de référence: {len(X_train)} profils de sol")
                    
                    # Créer et entraîner KNN
                    start_time = time.time()
                    self.agent.model = KNeighborsClassifier(**MODEL_PARAMS["knn"])
                    self.agent.model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    print(f"⏱️ [KNN] Indexation terminée en {training_time:.4f}s")
                    
                    # Prédictions
                    prediction_start = time.time()
                    y_pred = self.agent.model.predict(X_test)
                    y_pred_proba = self.agent.model.predict_proba(X_test)
                    prediction_time = time.time() - prediction_start
                    
                    # Métriques
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    print(f"📈 [KNN] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"🎯 [KNN] Precision: {precision:.4f}")
                    print(f"🔄 [KNN] Recall: {recall:.4f}")
                    print(f"⚖️ [KNN] F1-Score: {f1:.4f}")
                    print(f"⚡ [KNN] Prédiction en {prediction_time:.4f}s - Très rapide!")
                    
                    # Sauvegarder le modèle
                    joblib.dump(self.agent.model, "results/models/knn_model.pkl")
                    
                    # Préparer les résultats
                    self.agent.results = {
                        "model_name": "K-Nearest Neighbors",
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "training_time": float(training_time),
                        "prediction_time": float(prediction_time),
                        "y_test": y_test.tolist(),
                        "y_pred": y_pred.tolist(),
                        "y_pred_proba": y_pred_proba.tolist(),
                        "k_neighbors": k,
                        "feature_names": soil_data["feature_names"]
                    }
                    
                    # Envoyer résultats au comparateur
                    msg = Message(to="comparator@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.set_metadata("ontology", "model_results")
                    msg.body = json.dumps(self.agent.results)
                    await self.send(msg)
                    
                    print("📤 [KNN] Résultats envoyés au comparateur")
                    
                except Exception as e:
                    print(f"❌ [KNN] Erreur: {e}")
    
    async def setup(self):
        print("📍 [KNN] Agent KNN pour similarité des sols initialisé")
        template = Template()
        template.set_metadata("ontology", "soil_data")
        self.add_behaviour(self.TrainKNNBehaviour(), template) 
         