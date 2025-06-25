# ============================================================================
# FICHIER 15: utils/performance_metrics.py - Calculs métriques CORRIGÉ
# ============================================================================

"""
📊 Utilitaires pour calcul des métriques de performance agricole
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve)

class AgriculturalMetrics:
    """Métriques spécialisées pour l'analyse agricole"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculer toutes les métriques pour l'agriculture"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # AUC si probabilités disponibles
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                if y_pred_proba.ndim > 1:
                    proba_positive = y_pred_proba[:, 1]
                else:
                    proba_positive = y_pred_proba
                metrics['auc'] = float(roc_auc_score(y_true, proba_positive))
            except:
                metrics['auc'] = None
        
        return metrics
    
    @staticmethod
    def calculate_irrigation_efficiency(y_true, y_pred):
        """Calculer l'efficacité d'irrigation prédite"""
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Métriques spécifiques irrigation
            water_saved = tn  # Pas d'irrigation quand pas nécessaire
            unnecessary_irrigation = fp  # Irrigation inutile
            missed_irrigation = fn  # Irrigation manquée
            correct_irrigation = tp  # Irrigation correcte
            
            total_samples = len(y_true)
            
            efficiency_metrics = {
                'water_efficiency': float(water_saved / total_samples),
                'irrigation_precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                'irrigation_recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                'water_waste_rate': float(fp / total_samples),
                'drought_risk': float(fn / total_samples)
            }
            
            return efficiency_metrics
        
        return {}
    
    @staticmethod
    def agricultural_interpretation(metrics):
        """Interprétation agricole des métriques"""
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        interpretation = {
            'model_quality': 'Excellent' if accuracy > 0.9 else 
                           'Très bon' if accuracy > 0.85 else 
                           'Bon' if accuracy > 0.8 else 
                           'Acceptable' if accuracy > 0.75 else 'Insuffisant',
            
            'irrigation_reliability': 'Très fiable' if precision > 0.9 else
                                    'Fiable' if precision > 0.85 else
                                    'Modérément fiable' if precision > 0.8 else 'Peu fiable',
            
            'drought_prevention': 'Excellent' if recall > 0.9 else
                                'Très bon' if recall > 0.85 else
                                'Bon' if recall > 0.8 else 'Insuffisant',
            
            'practical_use': accuracy > 0.85,
            'recommendations': []
        }
        
        # Recommandations
        if accuracy > 0.85:
            interpretation['recommendations'].append("Modèle prêt pour usage pratique")
        if precision > 0.9:
            interpretation['recommendations'].append("Faible risque de sur-irrigation")
        if recall > 0.9:
            interpretation['recommendations'].append("Excellent pour prévenir le stress hydrique")
        if precision < 0.8:
            interpretation['recommendations'].append("Attention au gaspillage d'eau")
        if recall < 0.8:
            interpretation['recommendations'].append("Risque de manquer des besoins d'irrigation")
            
        return interpretation
    
    @staticmethod
    def crop_specific_analysis(y_true, y_pred, crop_labels=None):
        """Analyse spécifique par type de culture"""
        if crop_labels is None:
            return {}
        
        unique_crops = np.unique(crop_labels)
        crop_analysis = {}
        
        for crop in unique_crops:
            crop_mask = crop_labels == crop
            if np.sum(crop_mask) > 0:
                crop_y_true = y_true[crop_mask]
                crop_y_pred = y_pred[crop_mask]
                
                crop_metrics = AgriculturalMetrics.calculate_all_metrics(
                    crop_y_true, crop_y_pred
                )
                crop_analysis[crop] = crop_metrics
        
        return crop_analysis

