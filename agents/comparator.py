# ============================================================================
# FICHIER 12: agents/comparator_agent.py - Comparaison
# ============================================================================

import numpy as np
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
import json
import time

class AgroComparatorAgent(Agent):
    """📈 Agent comparateur des modèles agricoles"""
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.model_results = []
        self.models_received = 0
        self.total_models = 5
    
    class CompareModelsBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            
            if msg and msg.get_metadata("ontology") == "model_results":
                # Recevoir résultats d'un modèle
                model_result = json.loads(msg.body)
                self.agent.model_results.append(model_result)
                self.agent.models_received += 1
                
                model_name = model_result["model_name"]
                accuracy = model_result["accuracy"]
                print(f"📨 [COMPARATOR] Reçu {model_name}: {accuracy:.4f}")
                
                # Quand tous les modèles ont envoyé leurs résultats
                if self.agent.models_received >= self.agent.total_models:
                    print(f"\n📈 [COMPARATOR] Comparaison de {self.agent.total_models} modèles agricoles")
                    print("=" * 60)
                    
                    # Trier par accuracy
                    sorted_results = sorted(self.agent.model_results, 
                                          key=lambda x: x["accuracy"], reverse=True)
                    
                    # Afficher le classement
                    print("🏆 CLASSEMENT DES MODÈLES:")
                    for i, result in enumerate(sorted_results, 1):
                        name = result["model_name"]
                        acc = result["accuracy"]
                        time_taken = result["training_time"]
                        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                        print(f"{emoji} {i}. {name}")
                        print(f"    Accuracy: {acc:.4f} ({acc*100:.2f}%)")
                        print(f"    Temps: {time_taken:.2f}s")
                        print()
                    
                    # Statistiques globales
                    accuracies = [r["accuracy"] for r in self.agent.model_results]
                    times = [r["training_time"] for r in self.agent.model_results]
                    
                    print("📊 STATISTIQUES GLOBALES:")
                    print(f"Accuracy moyenne: {np.mean(accuracies):.4f}")
                    print(f"Accuracy max: {np.max(accuracies):.4f}")
                    print(f"Accuracy min: {np.min(accuracies):.4f}")
                    print(f"Temps total: {sum(times):.2f}s")
                    print(f"Temps moyen: {np.mean(times):.2f}s")
                    
                    # Recommandations agricoles
                    best_model = sorted_results[0]
                    fastest_model = min(self.agent.model_results, key=lambda x: x["training_time"])
                    
                    print("\n💡 RECOMMANDATIONS POUR L'AGRICULTURE:")
                    print(f"🏆 Meilleur modèle: {best_model['model_name']} ({best_model['accuracy']*100:.1f}%)")
                    print(f"⚡ Plus rapide: {fastest_model['model_name']} ({fastest_model['training_time']:.2f}s)")
                    
                    if best_model["accuracy"] > 0.85:
                        print("✅ Excellent pour prédiction d'irrigation")
                    elif best_model["accuracy"] > 0.80:
                        print("✅ Bon pour aide à la décision")
                    else:
                        print("⚠️ Nécessite amélioration pour usage pratique")
                    
                    # Préparer rapport complet
                    comparison_report = {
                        "models_compared": self.agent.total_models,
                        "best_model": best_model,
                        "fastest_model": fastest_model,
                        "all_results": sorted_results,
                        "statistics": {
                            "mean_accuracy": float(np.mean(accuracies)),
                            "max_accuracy": float(np.max(accuracies)),
                            "min_accuracy": float(np.min(accuracies)),
                            "total_time": float(sum(times)),
                            "mean_time": float(np.mean(times))
                        },
                        "timestamp": time.time()
                    }
                    
                    # Sauvegarder le rapport
                    with open("results/metrics/comparison_report.json", "w") as f:
                        json.dump(comparison_report, f, indent=2)
                    
                    # Envoyer au visualizer
                    msg = Message(to="visualizer@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.set_metadata("ontology", "comparison_results")
                    msg.body = json.dumps(comparison_report)
                    await self.send(msg)
                    
                    print("📤 [COMPARATOR] Rapport envoyé au visualizer")
    
    async def setup(self):
        print("📈 [COMPARATOR] Agent comparateur agricole initialisé")
        template = Template()
        template.set_metadata("ontology", "model_results")
        self.add_behaviour(self.CompareModelsBehaviour(), template)
  