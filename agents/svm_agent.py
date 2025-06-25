# ============================================================================
# FICHIER 7: agents/svm_agent.py - SVM CORRIGÉ
# ============================================================================

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
import json 
import time
import joblib
from config import MODEL_PARAMS

class SVMSoilAgent(Agent):
    """🔍 Agent SVM spécialisé pour analyse complexe des sols"""
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.model = None
        self.results = None
    
    class TrainSVMBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            
            if msg and msg.get_metadata("ontology") == "soil_data":
                print("\n🔍 [SVM] Analyse SVM des caractéristiques du sol")
                
                try:
                    # Décoder les données
                    soil_data = json.loads(msg.body)
                    X_train = np.array(soil_data["X_train"])
                    X_test = np.array(soil_data["X_test"])
                    y_train = np.array(soil_data["y_train"])
                    y_test = np.array(soil_data["y_test"])
                    
                    print(f"📊 [SVM] Analyse de {len(X_train)} profils de sol")
                    
                    # Créer et entraîner le modèle SVM
                    start_time = time.time()
                    self.agent.model = SVC(**MODEL_PARAMS["svm"], probability=True)
                    self.agent.model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    print(f"⏱️ [SVM] Analyse terminée en {training_time:.2f}s")
                    print(f"🎯 [SVM] Vecteurs support: {len(self.agent.model.support_)}")
                    
                    # Prédictions
                    y_pred = self.agent.model.predict(X_test)
                    y_pred_proba = self.agent.model.predict_proba(X_test)
                    
                    # Métriques
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    print(f"📈 [SVM] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"🎯 [SVM] Precision: {precision:.4f}")
                    print(f"🔄 [SVM] Recall: {recall:.4f}")
                    print(f"⚖️ [SVM] F1-Score: {f1:.4f}")
                    print(f"🔬 [SVM] Kernel: {self.agent.model.kernel}")
                    
                    # Sauvegarder le modèle
                    joblib.dump(self.agent.model, "results/models/svm_model.pkl")
                    
                    # Préparer les résultats
                    self.agent.results = {
                        "model_name": "Support Vector Machine",
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "training_time": float(training_time),
                        "y_test": y_test.tolist(),
                        "y_pred": y_pred.tolist(),
                        "y_pred_proba": y_pred_proba.tolist(),
                        "n_support_vectors": int(len(self.agent.model.support_)),
                        "kernel": self.agent.model.kernel,
                        "feature_names": soil_data["feature_names"]
                    }
                    
                    # Envoyer résultats au comparateur
                    msg = Message(to="comparator@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.set_metadata("ontology", "model_results")
                    msg.body = json.dumps(self.agent.results)
                    await self.send(msg)
                    
                    print("📤 [SVM] Résultats envoyés au comparateur")
                    
                except Exception as e:
                    print(f"❌ [SVM] Erreur: {e}")
    
    async def setup(self):
        print("🔍 [SVM] Agent SVM pour analyse des sols initialisé")
        template = Template()
        template.set_metadata("ontology", "soil_data")
        self.add_behaviour(self.TrainSVMBehaviour(), template)


