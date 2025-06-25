#!/usr/bin/env python3
"""
🛠️ Script de correction rapide pour votre projet
"""

import os

def main():
    print("🛠️ CORRECTION RAPIDE DU PROJET")
    print("=" * 40)
    
    # 1. Vérifier et créer les fichiers __init__.py manquants
    print("📝 Vérification des fichiers __init__.py...")
    
    init_files = {
        'agents/__init__.py': '''"""
🌾 Package des agents SPADE pour l'agriculture
Système multi-agent pour prédiction d'irrigation
"""

__version__ = "1.0.0"
__author__ = "Agricultural ML System"
''',
        'utils/__init__.py': '''"""
🔧 Utilitaires pour le système multi-agent agricole
"""

__version__ = "1.0.0"
'''
    }
    
    for file_path, content in init_files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ Créé: {file_path}")
        else:
            print(f"   ✅ Existe: {file_path}")
    
    # 2. Vérifier la structure results
    print("\n📁 Vérification structure results...")
    result_dirs = [
        'results/models',
        'results/metrics', 
        'results/plots',
        'results/reports'
    ]
    
    for directory in result_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    # 3. Vérifier les fichiers critiques
    print("\n🔍 Vérification des fichiers...")
    critical_files = [
        'main.py',
        'config.py', 
        'test_system.py',
        'datafinal1.csv'
    ]
    
    for file in critical_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MANQUANT")
    
    print("\n🎉 Correction terminée!")
    print("📋 Commandes recommandées:")
    print("   1. python test_system.py")
    print("   2. python main.py")

if __name__ == "__main__":
    main()
