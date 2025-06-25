
 ============================================================================
# FICHIER 12: agents/visualizer_agent.py - Visualisation CORRIGÉE
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
                    
                    # Configurer le style
                    plt.style.use('default')  # CORRIGÉ
                    
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
                    
                    # Sauvegarder le graphique
                    plt.savefig('results/plots/agricultural_models_comparison.png', 
                               dpi=300, bbox_inches='tight')
                    print("📊 [VISUALIZER] Graphique principal sauvegardé")
                    
                    # Créer graphique importance des features (si disponible)
                    self.create_feature_importance_plot(results)
                    
                    # Créer rapport HTML
                    self.create_html_report(comparison_data)
                    
                    # Créer rapport texte
                    self.create_text_report(comparison_data)
                    
                    plt.close('all')  # CORRIGÉ: Fermer les figures
                    
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
        
        def create_feature_importance_plot(self, results):
            """Créer graphique d'importance des features"""
            forest_result = None
            for result in results:
                if "feature_importance" in result:
                    forest_result = result
                    break
            
            if forest_result:
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
        
        def create_html_report(self, comparison_data):
            """Créer rapport HTML"""
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>🌾 Rapport d'Analyse Agricole</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #2E8B57; color: white; padding: 20px; text-align: center; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                    .best {{ background-color: #d4edda; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🌾 Rapport d'Analyse des Modèles Agricoles</h1>
                    <p>Prédiction des Besoins d'Irrigation - {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                </div>
                
                <div class="section best">
                    <h2>🏆 Meilleur Modèle</h2>
                    <h3>{comparison_data['best_model']['model_name']}</h3>
                    <div class="metric">
                        <strong>Accuracy:</strong> {comparison_data['best_model']['accuracy']:.4f}
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
                </div>
                
                <div class="section">
                    <h2>📊 Comparaison Complète</h2>
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
                html_content += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{result['model_name']}</td>
                            <td>{result['accuracy']:.4f}</td>
                            <td>{result['precision']:.4f}</td>
                            <td>{result['recall']:.4f}</td>
                            <td>{result['f1_score']:.4f}</td>
                            <td>{result['training_time']:.2f}</td>
                        </tr>"""
            
            html_content += """
                    </table>
                </div>
            </body>
            </html>"""
            
            with open('results/reports/agricultural_analysis_report.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print("🌐 [VISUALIZER] Rapport HTML créé")
        
        def create_text_report(self, comparison_data):
            """Créer rapport texte détaillé"""
            report = f"""
🌾 RAPPORT D'ANALYSE DES MODÈLES AGRICOLES
{'='*60}
Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Système Multi-Agent pour Prédiction d'Irrigation

🏆 CLASSEMENT FINAL:
{'-'*30}
"""
            
            for i, result in enumerate(comparison_data['all_results'], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                report += f"""
{emoji} {i}. {result['model_name']}
   Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)
   Precision: {result['precision']:.4f}
   Recall: {result['recall']:.4f}
   F1-Score: {result['f1_score']:.4f}
   Temps: {result['training_time']:.2f}s
"""
            
            stats = comparison_data['statistics']
            report += f"""

📈 STATISTIQUES GLOBALES:
{'-'*30}
Accuracy moyenne: {stats['mean_accuracy']:.4f}
Accuracy maximale: {stats['max_accuracy']:.4f}
Accuracy minimale: {stats['min_accuracy']:.4f}
Temps total: {stats['total_time']:.2f} secondes
Temps moyen: {stats['mean_time']:.2f} secondes

💡 RECOMMANDATIONS AGRICOLES:
{'-'*30}
• Le modèle {comparison_data['best_model']['model_name']} est recommandé pour la prédiction d'irrigation
• Accuracy de {comparison_data['best_model']['accuracy']*100:.1f}% indique une {'excellente' if comparison_data['best_model']['accuracy'] > 0.9 else 'très bonne' if comparison_data['best_model']['accuracy'] > 0.85 else 'bonne'} fiabilité
• Peut être utilisé pour optimiser l'irrigation et économiser l'eau
• Réduction potentielle du gaspillage d'eau de {(1-comparison_data['statistics']['mean_accuracy'])*100:.1f}%

🌱 IMPACT ENVIRONNEMENTAL:
{'-'*30}
• Optimisation de l'usage de l'eau agricole
• Amélioration du rendement des cultures
• Réduction de l'impact environnemental
• Support à l'agriculture de précision

📁 FICHIERS GÉNÉRÉS:
{'-'*30}
• results/plots/agricultural_models_comparison.png
• results/plots/feature_importance.png
• results/reports/agricultural_analysis_report.html
• results/models/ (modèles sauvegardés)
• results/metrics/comparison_report.json

{'='*60}
Fin du rapport - Système Multi-Agent Agricole
"""
            
            with open('results/reports/agricultural_analysis_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            
            print("📝 [VISUALIZER] Rapport texte créé")
    
    async def setup(self):
        print("🎨 [VISUALIZER] Agent visualiseur agricole initialisé")
        template = Template()
        template.set_metadata("ontology", "comparison_results")
        self.add_behaviour(self.CreateVisualizationsBehaviour(), template)
