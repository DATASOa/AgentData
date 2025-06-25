# ============================================================================
# FICHIER 6: agents/logistic_agent.py - Régression Logistique CORRIGÉE
# ============================================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
import json
import time
import joblib
from config import MODEL_PARAMS

class LogisticCropAgent(Agent):
    """🧠 Agent spécialisé en régression logistique pour cultures"""
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.model = None
        self.results = None
    
    class TrainLogisticBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            
            if msg and msg.get_metadata("ontology") == "soil_data":
                print("\n🧠 [LOGISTIC] Réception des données de sol")
                
                try:
                    # Décoder les données
                    soil_data = json.loads(msg.body)
                    X_train = np.array(soil_data["X_train"])
                    X_test = np.array(soil_data["X_test"])
                    y_train = np.array(soil_data["y_train"])
                    y_test = np.array(soil_data["y_test"])
                    
                    print(f"📊 [LOGISTIC] Entraînement sur {len(X_train)} échantillons de sol")
                    
                    # Créer et entraîner le modèle
                    start_time = time.time()
                    self.agent.model = LogisticRegression(**MODEL_PARAMS["logistic"])
                    self.agent.model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    print(f"⏱️ [LOGISTIC] Entraînement terminé en {training_time:.2f}s")
                    
                    # Prédictions
                    y_pred = self.agent.model.predict(X_test)
                    y_pred_proba = self.agent.model.predict_proba(X_test)
                    
                    # Métriques
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    print(f"📈 [LOGISTIC] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"🎯 [LOGISTIC] Precision: {precision:.4f}")
                    print(f"🔄 [LOGISTIC] Recall: {recall:.4f}")
                    print(f"⚖️ [LOGISTIC] F1-Score: {f1:.4f}")
                    
                    # Analyse des coefficients
                    feature_names = soil_data["feature_names"]
                    coefficients = self.agent.model.coef_[0]
                    
                    print("🧪 [LOGISTIC] Importance des paramètres du sol:")
                    coef_importance = [(feat, coef) for feat, coef in zip(feature_names, coefficients)]
                    coef_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    for i, (feature, coef) in enumerate(coef_importance[:3]):
                        print(f"   {i+1}. {feature}: {coef:.4f}")
                    
                    # Sauvegarder le modèle
                    joblib.dump(self.agent.model, "results/models/logistic_model.pkl")
                    
                    # Préparer les résultats
                    self.agent.results = {
                        "model_name": "Logistic Regression",
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "training_time": float(training_time),
                        "y_test": y_test.tolist(),
                        "y_pred": y_pred.tolist(),
                        "y_pred_proba": y_pred_proba.tolist(),
                        "coefficients": coefficients.tolist(),
                        "feature_names": feature_names,
                        "intercept": float(self.agent.model.intercept_[0])
                    }
                    
                    # Envoyer résultats au comparateur
                    msg = Message(to="comparator@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.set_metadata("ontology", "model_results")
                    msg.body = json.dumps(self.agent.results)
                    await self.send(msg)
                    
                    print("📤 [LOGISTIC] Résultats envoyés au comparateur")
                    
                except Exception as e:
                    print(f"❌ [LOGISTIC] Erreur: {e}")
    
    async def setup(self):
        print("🧠 [LOGISTIC] Agent régression logistique initialisé")
        template = Template()
        template.set_metadata("ontology", "soil_data")
        self.add_behaviour(self.TrainLogisticBehaviour(), template)


