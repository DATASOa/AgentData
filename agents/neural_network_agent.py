# ============================================================================
# FICHIER 11: agents/neural_network_agent.py - Réseau de Neurones
# ============================================================================

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
import json
import time
import joblib
from config import MODEL_PARAMS

class NeuralCropAgent(Agent):
    """🧬 Agent réseau de neurones pour patterns complexes des cultures"""
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.model = None
        self.results = None
     
    class TrainNeuralBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            
            if msg and msg.get_metadata("ontology") == "soil_data":
                print("\n🧬 [NEURAL] Apprentissage neuronal des patterns agricoles")
                
                try:
                    # Décoder les données
                    soil_data = json.loads(msg.body)
                    X_train = np.array(soil_data["X_train"])
                    X_test = np.array(soil_data["X_test"])
                    y_train = np.array(soil_data["y_train"])
                    y_test = np.array(soil_data["y_test"])
                    
                    layers = MODEL_PARAMS["neural_network"]["hidden_layer_sizes"]
                    print(f"🧠 [NEURAL] Architecture: {len(X_train[0])} → {layers[0]} → {layers[1]} → 1")
                    print(f"📊 [NEURAL] Entraînement sur {len(X_train)} échantillons")
                    
                    # Créer et entraîner le réseau de neurones
                    start_time = time.time()
                    self.agent.model = MLPClassifier(**MODEL_PARAMS["neural_network"])
                    self.agent.model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    print(f"⏱️ [NEURAL] Apprentissage terminé en {training_time:.2f}s")
                    print(f"🔄 [NEURAL] Itérations: {self.agent.model.n_iter_}")
                    
                    # Prédictions
                    y_pred = self.agent.model.predict(X_test)
                    y_pred_proba = self.agent.model.predict_proba(X_test)
                    
                    # Métriques
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    print(f"📈 [NEURAL] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"🎯 [NEURAL] Precision: {precision:.4f}")
                    print(f"🔄 [NEURAL] Recall: {recall:.4f}")
                    print(f"⚖️ [NEURAL] F1-Score: {f1:.4f}")
                    
                    convergence_status = "Convergé" if self.agent.model.n_iter_ < MODEL_PARAMS["neural_network"]["max_iter"] else "Max itérations"
                    print(f"✅ [NEURAL] Statut: {convergence_status}")
                    
                    # Sauvegarder le modèle
                    joblib.dump(self.agent.model, "results/models/neural_network_model.pkl")
                    
                    # Préparer les résultats
                    self.agent.results = {
                        "model_name": "Neural Network",
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "training_time": float(training_time),
                        "y_test": y_test.tolist(),
                        "y_pred": y_pred.tolist(),
                        "y_pred_proba": y_pred_proba.tolist(),
                        "n_iterations": int(self.agent.model.n_iter_),
                        "hidden_layers": layers,
                        "activation": MODEL_PARAMS["neural_network"]["activation"],
                        "feature_names": soil_data["feature_names"]
                    }
                    
                    # Envoyer résultats au comparateur
                    msg = Message(to="comparator@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.set_metadata("ontology", "model_results")
                    msg.body = json.dumps(self.agent.results)
                    await self.send(msg)
                    
                    print("📤 [NEURAL] Résultats envoyés au comparateur")
                    
                except Exception as e:
                    print(f"❌ [NEURAL] Erreur: {e}")
    
    async def setup(self):
        print("🧬 [NEURAL] Agent réseau de neurones initialisé")
        template = Template()
        template.set_metadata("ontology", "soil_data")
        self.add_behaviour(self.TrainNeuralBehaviour(), template)
 
