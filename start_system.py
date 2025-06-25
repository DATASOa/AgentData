# ============================================================================
# FICHIER 18: start_system.py - Script de démarrage SIMPLE
# ============================================================================

#!/usr/bin/env python3
"""
🚀 Script de démarrage simplifié du système multi-agent
"""
  
import os
import sys
import asyncio

def check_requirements():
    """Vérifier les prérequis"""
    print("🔍 Vérification des prérequis...")
    
    # Vérifier Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis")
        return False
    
    # Vérifier fichier de données
    if not os.path.exists('datafinal1.csv'):
        print("❌ Fichier 'datafinal1.csv' manquant")
        print("📁 Placez le fichier dans le répertoire principal")
        return False
    
    # Vérifier les imports
    try:
        import spade
        import pandas
        import sklearn
        import matplotlib
        import numpy
        print("✅ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("📦 Installez avec: pip install -r requirements.txt")
        return False

def setup_directories():
    """Créer les dossiers nécessaires"""
    print("📁 Création des dossiers...")
    
    directories = [
        'results',
        'results/models',
        'results/metrics', 
        'results/plots',
        'results/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Dossiers créés")

async def main():
    """Fonction principale"""
    print("🌾 SYSTÈME MULTI-AGENT AGRICOLE")
    print("=" * 50)
    
    # Vérifications
    if not check_requirements():
        return
    
    setup_directories()
    
    # Lancer le système principal
    try:
        print("\n🚀 Démarrage du système...")
        from main import main as run_system
        await run_system()
    except KeyboardInterrupt:
        print("\n⚠️ Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("\n🧪 Essayez d'abord: python test_system.py")

if __name__ == "__main__":
    asyncio.run(main())