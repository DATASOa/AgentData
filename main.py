# ============================================================================
# FICHIER 1: main.py - Point d'entrée principal
# ============================================================================

import asyncio
import sys
import os
from agents.coordinator_agent import CoordinatorAgent
from agents.data_manager_agent import SoilDataManagerAgent
from agents.logistic_agent import LogisticCropAgent
from agents.svm_agent import SVMSoilAgent
from agents.random_forest_agent import ForestCropAgent
from agents.knn_agent import KNNSoilAgent
from agents.neural_network_agent import NeuralCropAgent
from agents.comparator_agent import AgroComparatorAgent
from agents.visualizer_agent import AgroVisualizerAgent

async def main():
    """🌾 Système Multi-Agent pour Prédiction d'Irrigation Agricole"""
    
    print("🌾 SYSTÈME MULTI-AGENT AGRICOLE")
    print("🎯 Objectif: Prédire les besoins d'irrigation des cultures")
    print("=" * 60)
    
    # Vérifier que le fichier de données existe
    if not os.path.exists('data/datafinal1.csv'):
        print("❌ Erreur: fichier 'data/datafinal1.csv' non trouvé")
        print("📁 Placez le fichier dans le dossier data/")
        return
    
    # Créer les dossiers nécessaires
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    # Créer les agents
    agents = {
        'coordinator': CoordinatorAgent("coordinator@localhost", "agri2024"),
        'data_manager': SoilDataManagerAgent("soilmanager@localhost", "soil2024"),
        'logistic': LogisticCropAgent("logistic@localhost", "log2024"),
        'svm': SVMSoilAgent("svm@localhost", "svm2024"),
        'forest': ForestCropAgent("forest@localhost", "tree2024"),
        'knn': KNNSoilAgent("knn@localhost", "knn2024"),
        'neural': NeuralCropAgent("neural@localhost", "nn2024"),
        'comparator': AgroComparatorAgent("comparator@localhost", "comp2024"),
        'visualizer': AgroVisualizerAgent("visualizer@localhost", "viz2024")
    }
    
    print("🚀 Démarrage des agents...")
    
    try:
        # Démarrer tous les agents
        for name, agent in agents.items():
            await agent.start()
            print(f"✅ {name.title()} démarré")
        
        print("\n⏳ Système en cours d'exécution...")
        print("📊 Analyse des données agricoles...")
        
        # Attendre que le processus se termine
        while agents['coordinator'].is_alive():
            await asyncio.sleep(1)
        
        print("\n🎉 Analyse terminée avec succès !")
        print("📁 Consultez le dossier 'results/' pour les résultats")
        
    except KeyboardInterrupt:
        print("\n⚠️ Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
    finally:
        # Arrêter tous les agents
        for agent in agents.values():
            if agent.is_alive():
                await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())  