# ============================================================================
# FICHIER 14: utils/message_types.py - Types de messages
# ============================================================================

"""
🔄 Types de messages pour la communication entre agents
"""

# Types d'ontologies (sujets des messages)
MESSAGE_ONTOLOGIES = {
    "START_PROCESSING": "start_processing",
    "SOIL_DATA": "soil_data",
    "MODEL_RESULTS": "model_results",
    "COMPARISON_RESULTS": "comparison_results",
    "PROCESS_COMPLETE": "process_complete"
}

# Types de performatives (actions)
MESSAGE_PERFORMATIVES = {
    "REQUEST": "request",      # Demande d'action
    "INFORM": "inform",        # Information/données
    "CONFIRM": "confirm",      # Confirmation
    "FAILURE": "failure"       # Échec/erreur
}

# Structure des messages
class MessageStructure:
    """Structure standardisée des messages entre agents"""
    
    @staticmethod
    def create_data_message(data, ontology, performative="inform"):
        """Créer un message avec données"""
        return {
            "ontology": ontology,
            "performative": performative,
            "data": data,
            "timestamp": None  # Sera rempli lors de l'envoi
        }
    
    @staticmethod
    def create_status_message(status, details=""):
        """Créer un message de statut"""
        return {
            "ontology": "status",
            "performative": "inform",
            "status": status,
            "details": details,
            "timestamp": None
        }
  