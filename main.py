# ============================================================================
# FICHIER CORRIGÉ: main.py - Point d'entrée principal
# ============================================================================

import asyncio
import sys
import os
import warnings
import logging
warnings.filterwarnings('ignore')

# Désactiver les logs SPADE pour éviter les erreurs SSL
logging.getLogger('slixmpp').setLevel(logging.CRITICAL)
logging.getLogger('aioxmpp').setLevel(logging.CRITICAL)

def check_structure():
    """Vérifier la structure du projet"""
    print("🔍 Vérification de la structure...")
    
    # Vérifier les dossiers critiques
    required_dirs = ['agents', 'utils', 'results']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"   📁 Créé: {directory}/")
    
    # Vérifier les fichiers __init__.py
    init_files = ['agents/__init__.py', 'utils/__init__.py']
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w', encoding='utf-8') as f:
                module = init_file.split('/')[0]
                f.write(f'"""\n🌾 Module {module}\n"""\n__version__ = "1.0.0"\n')
            print(f"   📝 Créé: {init_file}")
    
    # Créer structure results
    result_dirs = ['results/models', 'results/metrics', 'results/plots', 'results/reports']
    for directory in result_dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Structure vérifiée")

async def main():
    """🌾 Système Multi-Agent pour Prédiction d'Irrigation Agricole"""
    
    print("🌾 SYSTÈME MULTI-AGENT AGRICOLE")
    print("🎯 Objectif: Prédire les besoins d'irrigation des cultures")
    print("=" * 60)
    
    # Vérifier et corriger la structure
    check_structure()
    
    # Vérifier que le fichier de données existe
    if not os.path.exists('datafinal1.csv'):
        print("❌ Erreur: fichier 'datafinal1.csv' non trouvé")
        print("📁 Placez le fichier dans le répertoire principal")
        print("🧪 Ou testez d'abord avec: python test_system.py")
        return
    
    # Import des agents (après vérification de structure)
    try:
        from agents.coordinator import CoordinatorAgent
        from agents.data_manager import SoilDataManagerAgent
        from agents.logistic_agent import LogisticCropAgent
        from agents.svm_agent import SVMSoilAgent
        from agents.random_forest_agent import ForestCropAgent
        from agents.knn_agent import KNNSoilAgent
        from agents.neural_network_agent import NeuralCropAgent
        from agents.comparator import AgroComparatorAgent
        from agents.visualizer_agent import AgroVisualizerAgent
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        print("🔧 Vérifiez les fichiers agents/")
        return
    
    # Créer les agents avec mots de passe uniformes
    agents = {
        'coordinator': CoordinatorAgent("coordinator@localhost", "password"),
        'data_manager': SoilDataManagerAgent("soilmanager@localhost", "password"),
        'logistic': LogisticCropAgent("logistic@localhost", "password"),
        'svm': SVMSoilAgent("svm@localhost", "password"),
        'forest': ForestCropAgent("forest@localhost", "password"),
        'knn': KNNSoilAgent("knn@localhost", "password"),
        'neural': NeuralCropAgent("neural@localhost", "password"),
        'comparator': AgroComparatorAgent("comparator@localhost", "password"),
        'visualizer': AgroVisualizerAgent("visualizer@localhost", "password")
    }
    
    print("🚀 Démarrage des agents...")
    
    try:
        # Démarrer tous les agents
        for name, agent in agents.items():
            await agent.start()
            print(f"✅ {name.title()} démarré")
            await asyncio.sleep(0.5)  # Petit délai entre démarrages
        
        print("\n⏳ Système en cours d'exécution...")
        print("📊 Analyse des données agricoles...")
        
        # Attendre que le coordinateur termine
        while agents['coordinator'].is_alive():
            await asyncio.sleep(1)
        
        print("\n🎉 Analyse terminée avec succès !")
        print("📁 Consultez le dossier 'results/' pour les résultats")
        
    except KeyboardInterrupt:
        print("\n⚠️ Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("🧪 Essayez d'abord avec: python test_system.py")
    finally:
        # Arrêter tous les agents proprement
        print("\n🛑 Arrêt des agents...")
        for name, agent in agents.items():
            try:
                if agent.is_alive():
                    await agent.stop()
            except:
                pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        print("🧪 Testez d'abord avec: python test_system.py")
