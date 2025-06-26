# ============================================================================
# FICHIER: agents/visualizer_agent.py - Agent de visualisation agricole
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
import json
import os
from datetime import datetime

class AgroVisualizerAgent(Agent):
    """🎨 Agent visualiseur pour rapports agricoles"""
    
    class CreateVisualizationsBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            
            if msg and msg.get_metadata("ontology") == "comparison_results":
                print("\n🎨 [VISUALIZER] Création des graphiques agricoles")
                
                try:
                    comparison_data = json.loads(msg.body)
                    results = comparison_data["all_results"]
                    
                    # Configurer le style matplotlib
                    plt.style.use('default')
                    
                    # Créer figure avec sous-graphiques
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle('🌾 Analyse Comparative des Modèles Agricoles', 
                                fontsize=16, fontweight='bold')
                    
                    # 1. Graphique accuracy
                    models = [r["model_name"] for r in results]
                    accuracies = [r["accuracy"] for r in results]
                    colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#9370DB']
                    
                    bars = axes[0,0].bar(models, accuracies, color=colors)
                    axes[0,0].set_title('📈 Accuracy des Modèles', fontweight='bold')
                    axes[0,0].set_ylabel('Accuracy')
                    axes[0,0].set_ylim(0, 1)
                    
                    # Ajouter valeurs sur les barres
                    for bar, acc in zip(bars, accuracies):
                        height = bar.get_height()
                        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                      f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    axes[0,0].tick_params(axis='x', rotation=45)
                    
                    # 2. Temps d'entraînement
                    times = [r["training_time"] for r in results]
                    bars2 = axes[0,1].bar(models, times, color=colors)
                    axes[0,1].set_title('⏱️ Temps d\'Entraînement', fontweight='bold')
                    axes[0,1].set_ylabel('Temps (secondes)')
                    
                    for bar, time_val in zip(bars2, times):
                        height = bar.get_height()
                        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                                      f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
                    
                    axes[0,1].tick_params(axis='x', rotation=45)
                    
                    # 3. Matrice de confusion du meilleur modèle
                    best_model = results[0]
                    y_test = np.array(best_model["y_test"])
                    y_pred = np.array(best_model["y_pred"])
                    
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                               xticklabels=['Pas d\'irrigation', 'Irrigation'],
                               yticklabels=['Pas d\'irrigation', 'Irrigation'],
                               ax=axes[1,0])
                    axes[1,0].set_title(f'🎯 Matrice de Confusion - {best_model["model_name"]}', 
                                       fontweight='bold')
                    axes[1,0].set_xlabel('Prédictions')
                    axes[1,0].set_ylabel('Réalité')
                    
                    # 4. Comparaison F1-Score
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
                    
                    # Sauvegarder le graphique principal
                    plt.savefig('results/plots/agricultural_models_comparison.png', 
                               dpi=300, bbox_inches='tight')
                    print("📊 [VISUALIZER] Graphique principal sauvegardé")
                    
                    # Créer graphique importance des features (si disponible)
                    self.create_feature_importance_plot(results)
                    
                    # Créer rapports
                    self.create_html_report(comparison_data)
                    self.create_text_report(comparison_data)
                    
                    plt.close('all')
                    
                    print("✅ [VISUALIZER] Tous les rapports générés")
                    print("📁 [VISUALIZER] Fichiers dans 'results/'")
                    
                    # Notifier le coordinateur que c'est terminé
                    msg = Message(to="coordinator@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.set_metadata("ontology", "process_complete")
                    msg.body = "visualization_complete"
                    await self.send(msg)
                    
                except Exception as e:
                    print(f"❌ [VISUALIZER] Erreur: {e}")
                    import traceback
                    traceback.print_exc()
        
        def create_feature_importance_plot(self, results):
            """Créer graphique d'importance des features"""
            forest_result = None
            for result in results:
                if "feature_importance" in result:
                    forest_result = result
                    break
            
            if forest_result:
                try:
                    plt.figure(figsize=(10, 6))
                    features = forest_result["feature_names"]
                    importance = forest_result["feature_importance"]
                    
                    # Trier par importance
                    sorted_idx = np.argsort(importance)[::-1]
                    
                    plt.bar(range(len(features)), [importance[i] for i in sorted_idx], 
                           color='forestgreen')
                    plt.title('🌿 Importance des Paramètres Agricoles (Random Forest)', 
                             fontweight='bold', fontsize=14)
                    plt.xlabel('Paramètres du Sol et Climat')
                    plt.ylabel('Importance')
                    plt.xticks(range(len(features)), [features[i] for i in sorted_idx], 
                              rotation=45, ha='right')
                    
                    plt.tight_layout()
                    plt.savefig('results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    print("🌿 [VISUALIZER] Graphique d'importance des features sauvegardé")
                except Exception as e:
                    print(f"⚠️ [VISUALIZER] Erreur importance features: {e}")
        
        def create_html_report(self, comparison_data):
            """Créer rapport HTML détaillé"""
            try:
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>🌾 Rapport d'Analyse Agricole</title>
                    <meta charset="UTF-8">
                    <style>
                        body {{ 
                            font-family: Arial, sans-serif; 
                            margin: 40px; 
                            background-color: #f8f9fa;
                        }}
                        .header {{ 
                            background-color: #2E8B57; 
                            color: white; 
                            padding: 20px; 
                            text-align: center; 
                            border-radius: 10px;
                            margin-bottom: 20px;
                        }}
                        .section {{ 
                            margin: 20px 0; 
                            padding: 15px; 
                            border: 1px solid #ddd; 
                            background-color: white;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }}
                        .best {{ 
                            background-color: #d4edda; 
                            border-color: #c3e6cb;
                        }}
                        .metric {{ 
                            display: inline-block; 
                            margin: 10px; 
                            padding: 10px; 
                            border: 1px solid #ccc; 
                            border-radius: 5px;
                            background-color: #f8f9fa;
                        }}
                        table {{ 
                            border-collapse: collapse; 
                            width: 100%; 
                            margin-top: 15px;
                        }}
                        th, td {{ 
                            border: 1px solid #ddd; 
                            padding: 12px; 
                            text-align: center; 
                        }}
                        th {{ 
                            background-color: #2E8B57; 
                            color: white;
                            font-weight: bold;
                        }}
                        tr:nth-child(even) {{
                            background-color: #f2f2f2;
                        }}
                        .model-name {{
                            font-weight: bold;
                            color: #2E8B57;
                        }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🌾 Rapport d'Analyse des Modèles Agricoles</h1>
                        <p>Prédiction des Besoins d'Irrigation - {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                        <p>Système Multi-Agent SPADE</p>
                    </div>
                    
                    <div class="section best">
                        <h2>🏆 Meilleur Modèle</h2>
                        <h3 class="model-name">{comparison_data['best_model']['model_name']}</h3>
                        <div class="metric">
                            <strong>Accuracy:</strong> {comparison_data['best_model']['accuracy']:.4f} ({comparison_data['best_model']['accuracy']*100:.2f}%)
                        </div>
                        <div class="metric">
                            <strong>Precision:</strong> {comparison_data['best_model']['precision']:.4f}
                        </div>
                        <div class="metric">
                            <strong>Recall:</strong> {comparison_data['best_model']['recall']:.4f}
                        </div>
                        <div class="metric">
                            <strong>F1-Score:</strong> {comparison_data['best_model']['f1_score']:.4f}
                        </div>
                        <div class="metric">
                            <strong>Temps d'entraînement:</strong> {comparison_data['best_model']['training_time']:.2f}s
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>📊 Comparaison Complète des Modèles</h2>
                        <table>
                            <tr>
                                <th>Rang</th>
                                <th>Modèle</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>Temps (s)</th>
                            </tr>"""
                
                for i, result in enumerate(comparison_data['all_results'], 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                    html_content += f"""
                            <tr>
                                <td>{emoji} {i}</td>
                                <td class="model-name">{result['model_name']}</td>
                                <td>{result['accuracy']:.4f}</td>
                                <td>{result['precision']:.4f}</td>
                                <td>{result['recall']:.4f}</td>
                                <td>{result['f1_score']:.4f}</td>
                                <td>{result['training_time']:.2f}</td>
                            </tr>"""
                
                stats = comparison_data['statistics']
                html_content += f"""
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>📈 Statistiques Globales</h2>
                        <div class="metric">
                            <strong>Accuracy moyenne:</strong> {stats['mean_accuracy']:.4f}
                        </div>
                        <div class="metric">
                            <strong>Accuracy maximale:</strong> {stats['max_accuracy']:.4f}
                        </div>
                        <div class="metric">
                            <strong>Accuracy minimale:</strong> {stats['min_accuracy']:.4f}
                        </div>
                        <div class="metric">
                            <strong>Temps total:</strong> {stats['total_time']:.2f}s
                        </div>
                        <div class="metric">
                            <strong>Temps moyen:</strong> {stats['mean_time']:.2f}s
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>💡 Recommandations Agricoles</h2>
                        <ul>
                            <li>🏆 <strong>Modèle recommandé:</strong> {comparison_data['best_model']['model_name']} avec {comparison_data['best_model']['accuracy']*100:.1f}% d'accuracy</li>
                            <li>💧 <strong>Optimisation irrigation:</strong> Réduction potentielle du gaspillage d'eau de {(1-comparison_data['statistics']['mean_accuracy'])*100:.1f}%</li>
                            <li>🌱 <strong>Impact environnemental:</strong> Support à l'agriculture de précision</li>
                            <li>⚡ <strong>Efficacité:</strong> Prédictions rapides pour aide à la décision</li>
                        </ul>
                    </div>
                    
                    <div class="section">
                        <h2>📁 Fichiers Générés</h2>
                        <ul>
                            <li>📊 results/plots/agricultural_models_comparison.png</li>
                            <li>🌿 results/plots/feature_importance.png</li>
                            <li>🌐 results/reports/agricultural_analysis_report.html</li>
                            <li>📝 results/reports/agricultural_analysis_report.txt</li>
                            <li>🤖 results/models/ (modèles sauvegardés)</li>
                            <li>📈 results/metrics/comparison_report.json</li>
                        </ul>
                    </div>
                    
                    <footer style="text-align: center; margin-top: 40px; color: #666;">
                        <p>Système Multi-Agent Agricole - Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
                    </footer>
                </body>
                </html>"""
                
                with open('results/reports/agricultural_analysis_report.html', 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                print("🌐 [VISUALIZER] Rapport HTML créé")
                
            except Exception as e:
                print(f"⚠️ [VISUALIZER] Erreur création HTML: {e}")
        
        def create_text_report(self, comparison_data):
            """Créer rapport texte détaillé"""
            try:
                report = f"""
🌾 RAPPORT D'ANALYSE DES MODÈLES AGRICOLES
{'='*70}
Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Système Multi-Agent SPADE pour Prédiction d'Irrigation

🏆 CLASSEMENT FINAL:
{'-'*35}
"""
                
                for i, result in enumerate(comparison_data['all_results'], 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                    report += f"""
{emoji} {i}. {result['model_name']}
   📈 Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)
   🎯 Precision: {result['precision']:.4f}
   🔄 Recall: {result['recall']:.4f}
   ⚖️ F1-Score: {result['f1_score']:.4f}
   ⏱️ Temps: {result['training_time']:.2f}s
"""
                
                stats = comparison_data['statistics']
                report += f"""

📈 STATISTIQUES GLOBALES:
{'-'*35}
📊 Accuracy moyenne: {stats['mean_accuracy']:.4f}
🎯 Accuracy maximale: {stats['max_accuracy']:.4f}
📉 Accuracy minimale: {stats['min_accuracy']:.4f}
⏱️ Temps total: {stats['total_time']:.2f} secondes
⏰ Temps moyen: {stats['mean_time']:.2f} secondes

💡 RECOMMANDATIONS AGRICOLES:
{'-'*35}
🏆 Meilleur modèle: {comparison_data['best_model']['model_name']}
📈 Accuracy: {comparison_data['best_model']['accuracy']*100:.1f}% - {'Excellente' if comparison_data['best_model']['accuracy'] > 0.9 else 'Très bonne' if comparison_data['best_model']['accuracy'] > 0.85 else 'Bonne'} fiabilité
💧 Optimisation irrigation: Réduction gaspillage de {(1-comparison_data['statistics']['mean_accuracy'])*100:.1f}%
🌱 Usage pratique: {'Recommandé' if comparison_data['best_model']['accuracy'] > 0.85 else 'Nécessite amélioration'}

🌱 IMPACT ENVIRONNEMENTAL:
{'-'*35}
💧 Optimisation de l'usage de l'eau agricole
📈 Amélioration du rendement des cultures  
🌍 Réduction de l'impact environnemental
🎯 Support à l'agriculture de précision
⚡ Aide à la décision en temps réel

📁 FICHIERS GÉNÉRÉS:
{'-'*35}
📊 results/plots/agricultural_models_comparison.png
🌿 results/plots/feature_importance.png  
🌐 results/reports/agricultural_analysis_report.html
📝 results/reports/agricultural_analysis_report.txt
🤖 results/models/ (modèles ML sauvegardés)
📈 results/metrics/comparison_report.json

🔬 DÉTAILS TECHNIQUES:
{'-'*35}
🧪 Modèles testés: {len(comparison_data['all_results'])}
📊 Algorithmes: Logistic Regression, SVM, Random Forest, KNN, Neural Network
🎯 Métriques: Accuracy, Precision, Recall, F1-Score
⏱️ Performance: Temps d'entraînement et prédiction mesurés
🔄 Validation: Train/Test split avec stratification

{'='*70}
🌾 Fin du rapport - Système Multi-Agent Agricole 🌾
{'='*70}
"""
                
                with open('results/reports/agricultural_analysis_report.txt', 'w', encoding='utf-8') as f:
                    f.write(report)
                
                print("📝 [VISUALIZER] Rapport texte créé")
                
            except Exception as e:
                print(f"⚠️ [VISUALIZER] Erreur création rapport texte: {e}")
    
    async def setup(self):
        print("🎨 [VISUALIZER] Agent visualiseur agricole initialisé")
        template = Template()
        template.set_metadata("ontology", "comparison_results")
        self.add_behaviour(self.CreateVisualizationsBehaviour(), template)
          