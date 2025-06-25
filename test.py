#!/usr/bin/env python3
"""
🧪 Test simple du preprocessing et Logistic Regression
Sans SPADE pour vérifier que les données sont correctes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def test_data_loading():
    """Test du chargement des données"""
    print("📂 Test du chargement des données...")
    
    try:
        # Charger les données
        data = pd.read_csv("data/data-final1.csv")
        print(f"✅ Données chargées: {data.shape}")
        print(f"📊 Colonnes: {list(data.columns)}")
        print(f"🎯 Distribution target:\n{data['besoin_irrigation'].value_counts()}")
        print(f"🌱 Types de cultures: {data['label'].nunique()}")
        
        return data
        
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return None

def test_preprocessing(data):
    """Test du preprocessing"""
    print("\n🔧 Test du preprocessing...")
    
    try:
        # Séparer features et target
        X = data.drop(['besoin_irrigation'], axis=1)
        y = data['besoin_irrigation']
        
        # Encoder la variable 'label'
        le = LabelEncoder()
        X['label_encoded'] = le.fit_transform(X['label'])
        X = X.drop(['label'], axis=1)
        
        print(f"✅ Label encoding terminé")
        print(f"📊 Features après encoding: {X.shape}")
        print(f"🔢 Colonnes finales: {list(X.columns)}")
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ Normalisation terminée")
        print(f"📊 Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler
        
    except Exception as e:
        print(f"❌ Erreur preprocessing: {e}")
        return None

def test_logistic_regression(X_train, X_test, y_train, y_test):
    """Test de la régression logistique"""
    print("\n🧠 Test de la Régression Logistique...")
    
    try:
        # Créer et entraîner le modèle
        model = LogisticRegression(max_iter=1000, random_state=42)
        
        print("🚀 Entraînement en cours...")
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Entraînement terminé!")
        print(f"🎯 Accuracy: {accuracy:.4f}")
        
        # Rapport détaillé
        print("\n📊 Rapport de classification:")
        print(classification_report(y_test, y_pred))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔍 Matrice de confusion:")
        print(cm)
        
        # Visualisation simple
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion - Logistic Regression')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.savefig('results/test_confusion_matrix.png')
        print("📈 Matrice sauvegardée: results/test_confusion_matrix.png")
        
        return model, accuracy
        
    except Exception as e:
        print(f"❌ Erreur Logistic Regression: {e}")
        return None, 0

def main():
    """Test principal"""
    print("🧪 TEST SIMPLE DU SYSTÈME ML")
    print("=" * 50)
    
    # Test 1: Chargement
    data = test_data_loading()
    if data is None:
        return
    
    # Test 2: Preprocessing  
    result = test_preprocessing(data)
    if result is None:
        return
    
    X_train, X_test, y_train, y_test, le, scaler = result
    
    # Test 3: Logistic Regression
    model, accuracy = test_logistic_regression(X_train, X_test, y_train, y_test)
    
    if model is not None:
        print(f"\n🎉 Test réussi! Accuracy finale: {accuracy:.4f}")
        
        # Sauvegarde des informations importantes
        print(f"\n📋 Informations importantes:")
        print(f"   • Dataset: {data.shape[0]} échantillons")
        print(f"   • Features: {X_train.shape[1]} variables")
        print(f"   • Classes: Pas besoin irrigation (0), Besoin irrigation (1)")
        print(f"   • Train: {len(y_train)}, Test: {len(y_test)}")
        print(f"   • Accuracy: {accuracy:.1%}")
        
        if accuracy > 0.8:
            print("✅ Excellentes performances! Le système est prêt pour SPADE.")
        elif accuracy > 0.7:
            print("✅ Bonnes performances! Le système fonctionne bien.")
        else:
            print("⚠️ Performances moyennes. Vérifiez les données.")
    
    else:
        print("❌ Test échoué. Vérifiez les erreurs ci-dessus.")

if __name__ == "__main__":
    import os
    
    # Créer le dossier results
    os.makedirs("results", exist_ok=True)
    
    # Lancer le test
    main() 