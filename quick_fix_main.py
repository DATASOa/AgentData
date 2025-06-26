#!/usr/bin/env python3

def fix_main():
    content = '''#!/usr/bin/env python3
"""
🌾 Système Multi-Agent Agricole SPADE - Version fonctionnelle
"""

import asyncio
import sys
import os
import warnings
import logging
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')
logging.getLogger('slixmpp').setLevel(logging.CRITICAL)

MODEL_PARAMS = {
    "logistic": {"max_iter": 1000, "random_state": 42, "solver": "lbfgs"},
    "svm": {"kernel": "rbf", "C": 1.0, "random_state": 42, "gamma": "scale"},
    "random_forest": {"n_estimators": 100, "random_state": 42, "max_depth": 10, "min_samples_split": 5},
    "knn": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
    "neural_network": {"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam", "max_iter": 300, "random_state": 42, "early_stopping": True, "validation_fraction": 0.1}
}

def load_and_preprocess_data():
    print("\\n📊 [DATA MANAGER] Début du traitement des données agricoles")
    
    try:
        from config import DATA_CONFIG
        
        print("📁 [DATA MANAGER] Chargement du fichier CSV...")
        data = pd.read_csv(DATA_CONFIG["file_path"])
        print(f"✅ [DATA MANAGER] {len(data)} échantillons de sols chargés")
        
        features = DATA_CONFIG["features"]
        target = DATA_CONFIG["target"]
        
        X = data[features]
        y = data[target]
        
        print(f"💧 [DATA MANAGER] Irrigation nécessaire: {y.sum()}/{len(y)} cultures")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=DATA_CONFIG["test_size"], 
            random_state=DATA_CONFIG["random_state"],
            stratify=y
        )
        
        print(f"📈 [DATA MANAGER] Entraînement: {len(X_train)} échantillons")
        print(f"📊 [DATA MANAGER] Test: {len(X_test)} échantillons")
        
        os.makedirs("results/models", exist_ok=True)
        joblib.dump(scaler, "results/models/soil_scaler.pkl")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": features,
            "target_name": target,
            "crops": data["label"].unique().tolist()
        }
        
    except Exception as e:
        print(f"❌ [DATA MANAGER] Erreur: {e}")
        return None

def train_ml_models(soil_data):
    print("\\n🤖 [COORDINATOR] Lancement des agents ML...")
    
    results = []
    X_train = soil_data["X_train"]
    X_test = soil_data["X_test"]
    y_train = soil_data["y_train"]
    y_test = soil_data["y_test"]
    
    # 1. Agent Logistic Regression
    print("\\n🧠 [LOGISTIC] Entraînement sur {} échantillons".format(len(X_train)))
    start_time = time.time()
    model_lr = LogisticRegression(**MODEL_PARAMS["logistic"])
    model_lr.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_lr.predict(X_test)
    y_pred_proba = model_lr.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [LOGISTIC] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_lr, "results/models/logistic_model.pkl")
    
    results.append({
        "model_name": "Logistic Regression",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    # 2. Agent SVM
    print("\\n🔍 [SVM] Analyse de {} profils de sol".format(len(X_train)))
    start_time = time.time()
    model_svm = SVC(**MODEL_PARAMS["svm"], probability=True)
    model_svm.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_svm.predict(X_test)
    y_pred_proba = model_svm.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [SVM] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_svm, "results/models/svm_model.pkl")
    
    results.append({
        "model_name": "Support Vector Machine",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    # 3. Agent Random Forest
    print("\\n🌳 [FOREST] Entraînement de {} arbres".format(MODEL_PARAMS["random_forest"]["n_estimators"]))
    start_time = time.time()
    model_rf = RandomForestClassifier(**MODEL_PARAMS["random_forest"])
    model_rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_rf.predict(X_test)
    y_pred_proba = model_rf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [FOREST] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_rf, "results/models/random_forest_model.pkl")
    
    results.append({
        "model_name": "Random Forest",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_importance": model_rf.feature_importances_.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    # 4. Agent KNN
    print("\\n📍 [KNN] Recherche des {} sols similaires".format(MODEL_PARAMS["knn"]["n_neighbors"]))
    start_time = time.time()
    model_knn = KNeighborsClassifier(**MODEL_PARAMS["knn"])
    model_knn.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_knn.predict(X_test)
    y_pred_proba = model_knn.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [KNN] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_knn, "results/models/knn_model.pkl")
    
    results.append({
        "model_name": "K-Nearest Neighbors",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    # 5. Agent Neural Network
    print("\\n🧬 [NEURAL] Apprentissage neuronal")
    start_time = time.time()
    model_nn = MLPClassifier(**MODEL_PARAMS["neural_network"])
    model_nn.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_nn.predict(X_test)
    y_pred_proba = model_nn.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [NEURAL] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_nn, "results/models/neural_network_model.pkl")
    
    results.append({
        "model_name": "Neural Network",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    return results

def compare_models(results):
    print(f"\\n📈 [COMPARATOR] Comparaison de {len(results)} modèles agricoles")
    print("=" * 60)
    
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
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
    
    accuracies = [r["accuracy"] for r in results]
    times = [r["training_time"] for r in results]
    
    print("📊 STATISTIQUES GLOBALES:")
    print(f"Accuracy moyenne: {np.mean(accuracies):.4f}")
    print(f"Accuracy max: {np.max(accuracies):.4f}")
    print(f"Temps total: {sum(times):.2f}s")
    
    best_model = sorted_results[0]
    print("\\n💡 RECOMMANDATIONS POUR L'AGRICULTURE:")
    print(f"🏆 Meilleur modèle: {best_model['model_name']} ({best_model['accuracy']*100:.1f}%)")
    
    if best_model["accuracy"] > 0.85:
        print("✅ Excellent pour prédiction d'irrigation")
    elif best_model["accuracy"] > 0.80:
        print("✅ Bon pour aide à la décision")
    else:
        print("⚠️ Nécessite amélioration pour usage pratique")
    
    comparison_report = {
        "models_compared": len(results),
        "best_model": best_model,
        "all_results": sorted_results,
        "statistics": {
            "mean_accuracy": float(np.mean(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "total_time": float(sum(times)),
            "mean_time": float(np.mean(times))
        }
    }
    
    with open("results/metrics/comparison_report.json", "w") as f:
        json.dump(comparison_report, f, indent=2)
    
    print("📤 [COMPARATOR] Rapport envoyé au visualizer")
    
    return comparison_report

def create_visualizations(comparison_data):
    print("\\n🎨 [VISUALIZER] Création des graphiques agricoles")
    
    try:
        results = comparison_data["all_results"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🌾 Analyse Comparative des Modèles Agricoles SPADE', 
                    fontsize=16, fontweight='bold')
        
        models = [r["model_name"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#9370DB']
        
        bars = axes[0,0].bar(models, accuracies, color=colors[:len(models)])
        axes[0,0].set_title('📈 Accuracy des Modèles', fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[0,0].tick_params(axis='x', rotation=45)
        
        times = [r["training_time"] for r in results]
        bars2 = axes[0,1].bar(models, times, color=colors[:len(models)])
        axes[0,1].set_title('⏱️ Temps d\\'Entraînement', fontweight='bold')
        axes[0,1].set_ylabel('Temps (secondes)')
        
        for bar, time_val in zip(bars2, times):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                          f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        axes[0,1].tick_params(axis='x', rotation=45)
        
        best_model = results[0]
        y_test = np.array(best_model["y_test"])
        y_pred = np.array(best_model["y_pred"])
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Pas d\\'irrigation', 'Irrigation'],
                   yticklabels=['Pas d\\'irrigation', 'Irrigation'],
                   ax=axes[1,0])
        axes[1,0].set_title(f'🎯 Matrice de Confusion - {best_model["model_name"]}', 
                           fontweight='bold')
        axes[1,0].set_xlabel('Prédictions')
        axes[1,0].set_ylabel('Réalité')
        
        f1_scores = [r["f1_score"] for r in results]
        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[1,1].bar(x - width, f1_scores, width, label='F1-Score', color='#2E8B57')
        axes[1,1].bar(x, precisions, width, label='Precision', color='#4682B4')
        axes[1,1].bar(x + width, recalls, width, label='Recall', color='#DAA520')
        
        axes[1,1].set_title('📊 Métriques Détaillées', fontweight='bold')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(models, rotation=45)
        axes[1,1].legend()
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        plt.savefig('results/plots/agricultural_models_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("📊 [VISUALIZER] Graphique principal sauvegardé")
        
        plt.close('all')
        
        best_model = comparison_data["best_model"]
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🌾 Rapport Agricole SPADE</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
                .header {{ background-color: #2E8B57; color: white; padding: 20px; text-align: center; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; background-color: white; border-radius: 8px; }}
                .best {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌾 Rapport d'Analyse Agricole SPADE</h1>
                <p>Système Multi-Agent - {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            </div>
            
            <div class="section best">
                <h2>🏆 Meilleur Modèle</h2>
                <h3>{best_model['model_name']}</h3>
                <div class="metric">
                    <strong>Accuracy:</strong> {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)
                </div>
                <div class="metric">
                    <strong>Precision:</strong> {best_model['precision']:.4f}
                </div>
                <div class="metric">
                    <strong>Recall:</strong> {best_model['recall']:.4f}
                </div>
                <div class="metric">
                    <strong>F1-Score:</strong> {best_model['f1_score']:.4f}
                </div>
            </div>
            
            <div class="section">
                <h2>💡 Recommandations Agricoles</h2>
                <ul>
                    <li>🏆 Système Multi-Agent SPADE pour l'agriculture de précision</li>
                    <li>💧 Optimisation de l'irrigation basée sur l'IA</li>
                    <li>🌱 Réduction de l'impact environnemental</li>
                </ul>
            </div>
        </body>
        </html>"""
        
        with open('results/reports/agricultural_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("🌐 [VISUALIZER] Rapport HTML créé")
        print("✅ [VISUALIZER] Tous les rapports générés")
        print("📁 [VISUALIZER] Fichiers dans 'results/'")
        
    except Exception as e:
        print(f"⚠️ [VISUALIZER] Erreur visualisation: {e}")

async def main():
    print("🌾 SYSTÈME MULTI-AGENT AGRICOLE SPADE")
    print("🎯 Objectif: Prédire les besoins d'irrigation des cultures")
    print("=" * 60)
    
    start_time = time.time()
    
    if not os.path.exists('datafinal1.csv'):
        print("❌ Erreur: fichier 'datafinal1.csv' non trouvé")
        return
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    try:
        print("\\n🎪 [COORDINATOR] Démarrage du système multi-agent SPADE")
        
        soil_data = load_and_preprocess_data()
        if not soil_data:
            return
        
        results = train_ml_models(soil_data)
        if not results:
            return
        
        comparison_data = compare_models(results)
        create_visualizations(comparison_data)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\\n🎉 [COORDINATOR] Processus SPADE terminé avec succès!")
        print(f"⏱️ [COORDINATOR] Temps total: {total_time:.2f} secondes")
        
        best_model = comparison_data["best_model"]
        print(f"\\n🏆 MEILLEUR MODÈLE: {best_model['model_name']}")
        print(f"   📈 Accuracy: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.1f}%)")
        
        print("\\n📋 FICHIERS GÉNÉRÉS:")
        print("   📊 results/plots/agricultural_models_comparison.png")
        print("   🌐 results/reports/agricultural_analysis_report.html") 
        print("   📈 results/metrics/comparison_report.json")
        print("   🤖 results/models/ (modèles ML sauvegardés)")
        
    except Exception as e:
        print(f"\\n❌ Erreur: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\\n💥 Erreur: {e}")
'''
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ main.py corrigé créé")

if __name__ == "__main__":
    print("🔧 CORRECTION RAPIDE DU MAIN.PY")
    print("=" * 40)
    fix_main()
    print("🎉 Correction terminée!")
    print("🚀 Testez: python main.py")
EOF

python quick_fix_main.pycat > quick_fix_main.py << 'EOF'
#!/usr/bin/env python3

def fix_main():
    content = '''#!/usr/bin/env python3
"""
🌾 Système Multi-Agent Agricole SPADE - Version fonctionnelle
"""

import asyncio
import sys
import os
import warnings
import logging
import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')
logging.getLogger('slixmpp').setLevel(logging.CRITICAL)

MODEL_PARAMS = {
    "logistic": {"max_iter": 1000, "random_state": 42, "solver": "lbfgs"},
    "svm": {"kernel": "rbf", "C": 1.0, "random_state": 42, "gamma": "scale"},
    "random_forest": {"n_estimators": 100, "random_state": 42, "max_depth": 10, "min_samples_split": 5},
    "knn": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
    "neural_network": {"hidden_layer_sizes": (100, 50), "activation": "relu", "solver": "adam", "max_iter": 300, "random_state": 42, "early_stopping": True, "validation_fraction": 0.1}
}

def load_and_preprocess_data():
    print("\\n📊 [DATA MANAGER] Début du traitement des données agricoles")
    
    try:
        from config import DATA_CONFIG
        
        print("📁 [DATA MANAGER] Chargement du fichier CSV...")
        data = pd.read_csv(DATA_CONFIG["file_path"])
        print(f"✅ [DATA MANAGER] {len(data)} échantillons de sols chargés")
        
        features = DATA_CONFIG["features"]
        target = DATA_CONFIG["target"]
        
        X = data[features]
        y = data[target]
        
        print(f"💧 [DATA MANAGER] Irrigation nécessaire: {y.sum()}/{len(y)} cultures")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=DATA_CONFIG["test_size"], 
            random_state=DATA_CONFIG["random_state"],
            stratify=y
        )
        
        print(f"📈 [DATA MANAGER] Entraînement: {len(X_train)} échantillons")
        print(f"📊 [DATA MANAGER] Test: {len(X_test)} échantillons")
        
        os.makedirs("results/models", exist_ok=True)
        joblib.dump(scaler, "results/models/soil_scaler.pkl")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": features,
            "target_name": target,
            "crops": data["label"].unique().tolist()
        }
        
    except Exception as e:
        print(f"❌ [DATA MANAGER] Erreur: {e}")
        return None

def train_ml_models(soil_data):
    print("\\n🤖 [COORDINATOR] Lancement des agents ML...")
    
    results = []
    X_train = soil_data["X_train"]
    X_test = soil_data["X_test"]
    y_train = soil_data["y_train"]
    y_test = soil_data["y_test"]
    
    # 1. Agent Logistic Regression
    print("\\n🧠 [LOGISTIC] Entraînement sur {} échantillons".format(len(X_train)))
    start_time = time.time()
    model_lr = LogisticRegression(**MODEL_PARAMS["logistic"])
    model_lr.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_lr.predict(X_test)
    y_pred_proba = model_lr.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [LOGISTIC] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_lr, "results/models/logistic_model.pkl")
    
    results.append({
        "model_name": "Logistic Regression",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    # 2. Agent SVM
    print("\\n🔍 [SVM] Analyse de {} profils de sol".format(len(X_train)))
    start_time = time.time()
    model_svm = SVC(**MODEL_PARAMS["svm"], probability=True)
    model_svm.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_svm.predict(X_test)
    y_pred_proba = model_svm.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [SVM] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_svm, "results/models/svm_model.pkl")
    
    results.append({
        "model_name": "Support Vector Machine",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    # 3. Agent Random Forest
    print("\\n🌳 [FOREST] Entraînement de {} arbres".format(MODEL_PARAMS["random_forest"]["n_estimators"]))
    start_time = time.time()
    model_rf = RandomForestClassifier(**MODEL_PARAMS["random_forest"])
    model_rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_rf.predict(X_test)
    y_pred_proba = model_rf.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [FOREST] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_rf, "results/models/random_forest_model.pkl")
    
    results.append({
        "model_name": "Random Forest",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_importance": model_rf.feature_importances_.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    # 4. Agent KNN
    print("\\n📍 [KNN] Recherche des {} sols similaires".format(MODEL_PARAMS["knn"]["n_neighbors"]))
    start_time = time.time()
    model_knn = KNeighborsClassifier(**MODEL_PARAMS["knn"])
    model_knn.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_knn.predict(X_test)
    y_pred_proba = model_knn.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [KNN] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_knn, "results/models/knn_model.pkl")
    
    results.append({
        "model_name": "K-Nearest Neighbors",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    # 5. Agent Neural Network
    print("\\n🧬 [NEURAL] Apprentissage neuronal")
    start_time = time.time()
    model_nn = MLPClassifier(**MODEL_PARAMS["neural_network"])
    model_nn.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model_nn.predict(X_test)
    y_pred_proba = model_nn.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"📈 [NEURAL] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model_nn, "results/models/neural_network_model.pkl")
    
    results.append({
        "model_name": "Neural Network",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": soil_data["feature_names"]
    })
    
    return results

def compare_models(results):
    print(f"\\n📈 [COMPARATOR] Comparaison de {len(results)} modèles agricoles")
    print("=" * 60)
    
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
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
    
    accuracies = [r["accuracy"] for r in results]
    times = [r["training_time"] for r in results]
    
    print("📊 STATISTIQUES GLOBALES:")
    print(f"Accuracy moyenne: {np.mean(accuracies):.4f}")
    print(f"Accuracy max: {np.max(accuracies):.4f}")
    print(f"Temps total: {sum(times):.2f}s")
    
    best_model = sorted_results[0]
    print("\\n💡 RECOMMANDATIONS POUR L'AGRICULTURE:")
    print(f"🏆 Meilleur modèle: {best_model['model_name']} ({best_model['accuracy']*100:.1f}%)")
    
    if best_model["accuracy"] > 0.85:
        print("✅ Excellent pour prédiction d'irrigation")
    elif best_model["accuracy"] > 0.80:
        print("✅ Bon pour aide à la décision")
    else:
        print("⚠️ Nécessite amélioration pour usage pratique")
    
    comparison_report = {
        "models_compared": len(results),
        "best_model": best_model,
        "all_results": sorted_results,
        "statistics": {
            "mean_accuracy": float(np.mean(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "total_time": float(sum(times)),
            "mean_time": float(np.mean(times))
        }
    }
    
    with open("results/metrics/comparison_report.json", "w") as f:
        json.dump(comparison_report, f, indent=2)
    
    print("📤 [COMPARATOR] Rapport envoyé au visualizer")
    
    return comparison_report

def create_visualizations(comparison_data):
    print("\\n🎨 [VISUALIZER] Création des graphiques agricoles")
    
    try:
        results = comparison_data["all_results"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🌾 Analyse Comparative des Modèles Agricoles SPADE', 
                    fontsize=16, fontweight='bold')
        
        models = [r["model_name"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#9370DB']
        
        bars = axes[0,0].bar(models, accuracies, color=colors[:len(models)])
        axes[0,0].set_title('📈 Accuracy des Modèles', fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[0,0].tick_params(axis='x', rotation=45)
        
        times = [r["training_time"] for r in results]
        bars2 = axes[0,1].bar(models, times, color=colors[:len(models)])
        axes[0,1].set_title('⏱️ Temps d\\'Entraînement', fontweight='bold')
        axes[0,1].set_ylabel('Temps (secondes)')
        
        for bar, time_val in zip(bars2, times):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                          f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        axes[0,1].tick_params(axis='x', rotation=45)
        
        best_model = results[0]
        y_test = np.array(best_model["y_test"])
        y_pred = np.array(best_model["y_pred"])
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Pas d\\'irrigation', 'Irrigation'],
                   yticklabels=['Pas d\\'irrigation', 'Irrigation'],
                   ax=axes[1,0])
        axes[1,0].set_title(f'🎯 Matrice de Confusion - {best_model["model_name"]}', 
                           fontweight='bold')
        axes[1,0].set_xlabel('Prédictions')
        axes[1,0].set_ylabel('Réalité')
        
        f1_scores = [r["f1_score"] for r in results]
        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[1,1].bar(x - width, f1_scores, width, label='F1-Score', color='#2E8B57')
        axes[1,1].bar(x, precisions, width, label='Precision', color='#4682B4')
        axes[1,1].bar(x + width, recalls, width, label='Recall', color='#DAA520')
        
        axes[1,1].set_title('📊 Métriques Détaillées', fontweight='bold')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(models, rotation=45)
        axes[1,1].legend()
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        plt.savefig('results/plots/agricultural_models_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("📊 [VISUALIZER] Graphique principal sauvegardé")
        
        plt.close('all')
        
        best_model = comparison_data["best_model"]
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🌾 Rapport Agricole SPADE</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
                .header {{ background-color: #2E8B57; color: white; padding: 20px; text-align: center; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; background-color: white; border-radius: 8px; }}
                .best {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌾 Rapport d'Analyse Agricole SPADE</h1>
                <p>Système Multi-Agent - {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            </div>
            
            <div class="section best">
                <h2>🏆 Meilleur Modèle</h2>
                <h3>{best_model['model_name']}</h3>
                <div class="metric">
                    <strong>Accuracy:</strong> {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)
                </div>
                <div class="metric">
                    <strong>Precision:</strong> {best_model['precision']:.4f}
                </div>
                <div class="metric">
                    <strong>Recall:</strong> {best_model['recall']:.4f}
                </div>
                <div class="metric">
                    <strong>F1-Score:</strong> {best_model['f1_score']:.4f}
                </div>
            </div>
            
            <div class="section">
                <h2>💡 Recommandations Agricoles</h2>
                <ul>
                    <li>🏆 Système Multi-Agent SPADE pour l'agriculture de précision</li>
                    <li>💧 Optimisation de l'irrigation basée sur l'IA</li>
                    <li>🌱 Réduction de l'impact environnemental</li>
                </ul>
            </div>
        </body>
        </html>"""
        
        with open('results/reports/agricultural_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("🌐 [VISUALIZER] Rapport HTML créé")
        print("✅ [VISUALIZER] Tous les rapports générés")
        print("📁 [VISUALIZER] Fichiers dans 'results/'")
        
    except Exception as e:
        print(f"⚠️ [VISUALIZER] Erreur visualisation: {e}")

async def main():
    print("🌾 SYSTÈME MULTI-AGENT AGRICOLE SPADE")
    print("🎯 Objectif: Prédire les besoins d'irrigation des cultures")
    print("=" * 60)
    
    start_time = time.time()
    
    if not os.path.exists('datafinal1.csv'):
        print("❌ Erreur: fichier 'datafinal1.csv' non trouvé")
        return
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    try:
        print("\\n🎪 [COORDINATOR] Démarrage du système multi-agent SPADE")
        
        soil_data = load_and_preprocess_data()
        if not soil_data:
            return
        
        results = train_ml_models(soil_data)
        if not results:
            return
        
        comparison_data = compare_models(results)
        create_visualizations(comparison_data)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\\n🎉 [COORDINATOR] Processus SPADE terminé avec succès!")
        print(f"⏱️ [COORDINATOR] Temps total: {total_time:.2f} secondes")
        
        best_model = comparison_data["best_model"]
        print(f"\\n🏆 MEILLEUR MODÈLE: {best_model['model_name']}")
        print(f"   📈 Accuracy: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.1f}%)")
        
        print("\\n📋 FICHIERS GÉNÉRÉS:")
        print("   📊 results/plots/agricultural_models_comparison.png")
        print("   🌐 results/reports/agricultural_analysis_report.html") 
        print("   📈 results/metrics/comparison_report.json")
        print("   🤖 results/models/ (modèles ML sauvegardés)")
        
    except Exception as e:
        print(f"\\n❌ Erreur: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\\n💥 Erreur: {e}")
'''
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ main.py corrigé créé")

if __name__ == "__main__":
    print("🔧 CORRECTION RAPIDE DU MAIN.PY")
    print("=" * 40)
    fix_main()
    print("🎉 Correction terminée!")
    print("🚀 Testez: python main.py")
EOF

python quick_fix_main.py
