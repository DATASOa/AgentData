# ============================================================================
# FICHIER 9: agents/random_forest_agent.py - Random Forest
# ============================================================================

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
import json  
import time
import joblib
from config import MODEL_PARAMS

class ForestCropAgent(Agent):
    """🌳 Agent Random Forest pour analyse ensemble des cultures"""
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.model = None
        self.results = None
    
    class TrainForestBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            
            if msg and msg.get_metadata("ontology") == "soil_data":
                print("\n🌳 [FOREST] Analyse Random Forest des cultures")
                
                try:
                    # Décoder les données
                    soil_data = json.loads(msg.body)
                    X_train = np.array(soil_data["X_train"])
                    X_test = np.array(soil_data["X_test"])
                    y_train = np.array(soil_data["y_train"])
                    y_test = np.array(soil_data["y_test"])
                    
                    print(f"🌱 [FOREST] Entraînement de {MODEL_PARAMS['random_forest']['n_estimators']} arbres")
                    print(f"📊 [FOREST] Sur {len(X_train)} échantillons de cultures")
                    
                    # Créer et entraîner Random Forest
                    start_time = time.time()
                    self.agent.model = RandomForestClassifier(**MODEL_PARAMS["random_forest"])
                    self.agent.model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    print(f"⏱️ [FOREST] Forêt entraînée en {training_time:.2f}s")
                    
                    # Prédictions
                    y_pred = self.agent.model.predict(X_test)
                    y_pred_proba = self.agent.model.predict_proba(X_test)
                    
                    # Métriques
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    print(f"📈 [FOREST] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    print(f"🎯 [FOREST] Precision: {precision:.4f}")
                    print(f"🔄 [FOREST] Recall: {recall:.4f}")
                    print(f"⚖️ [FOREST] F1-Score: {f1:.4f}")
                    
                    # Importance des features
                    feature_names = soil_data["feature_names"]
                    feature_importance = self.agent.model.feature_importances_
                    
                    print("🌿 [FOREST] Importance des paramètres agricoles:")
                    importance_pairs = [(name, imp) for name, imp in zip(feature_names, feature_importance)]
                    importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (feature, importance) in enumerate(importance_pairs[:3]):
                        print(f"   {i+1}. {feature}: {importance:.4f} ({importance*100:.1f}%)")
                    
                    # Sauvegarder le modèle
                    joblib.dump(self.agent.model, "results/models/random_forest_model.pkl")
                    
                    # Préparer les résultats
                    self.agent.results = {
                        "model_name": "Random Forest",
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "training_time": float(training_time),
                        "y_test": y_test.tolist(),
                        "y_pred": y_pred.tolist(),
                        "y_pred_proba": y_pred_proba.tolist(),
                        "feature_importance": feature_importance.tolist(),
                        "feature_names": feature_names,
                        "n_estimators": MODEL_PARAMS["random_forest"]["n_estimators"]
                    }
                    
                    # Envoyer résultats au comparateur
                    msg = Message(to="comparator@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.set_metadata("ontology", "model_results")
                    msg.body = json.dumps(self.agent.results)
                    await self.send(msg)
                    
                    print("📤 [FOREST] Résultats envoyés au comparateur")
                    
                except Exception as e:
                    print(f"❌ [FOREST] Erreur: {e}")
    
    async def setup(self):
        print("🌳 [FOREST] Agent Random Forest pour cultures initialisé")
        template = Template()
        template.set_metadata("ontology", "soil_data")
        self.add_behaviour(self.TrainForestBehaviour(), template)