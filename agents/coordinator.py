# ============================================================================
# FICHIER 4: agents/coordinator.py - Coordinateur CORRIGÉ
# ============================================================================

import asyncio
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
import json
import time

class CoordinatorAgent(Agent):
    """🎪 Agent coordinateur principal du système agricole"""
    
    class CoordinateBehaviour(OneShotBehaviour):
        async def run(self):
            print("\n🎪 [COORDINATOR] Démarrage du système multi-agent")
            print("=" * 50)
            
            start_time = time.time()
            
            # Étape 1: Démarrer le gestionnaire de données
            print("📊 [COORDINATOR] Démarrage de l'analyse des sols...")
            await asyncio.sleep(2)  # Laisser temps aux autres agents de démarrer
            
            # Envoyer signal de démarrage au data manager
            msg = Message(to="soilmanager@localhost")
            msg.set_metadata("performative", "request")
            msg.set_metadata("ontology", "start_processing")
            msg.body = json.dumps({"action": "load_soil_data", "timestamp": time.time()})
            await self.send(msg)
            
            print("✅ [COORDINATOR] Signal envoyé au gestionnaire de données")
            
            # Attendre la fin du processus (sera notifié par le visualizer)
            print("⏳ [COORDINATOR] Supervision en cours...")
            
            # Le coordinator s'arrête quand le visualizer termine
            while True:
                msg = await self.receive(timeout=60)
                if msg and msg.get_metadata("ontology") == "process_complete":
                    break
                await asyncio.sleep(1)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n🎉 [COORDINATOR] Processus terminé en {total_time:.2f} secondes")
            print("📁 [COORDINATOR] Résultats disponibles dans 'results/'")
            
            # Arrêter le système
            await asyncio.sleep(2)
            await self.agent.stop()
    
    async def setup(self):
        print("🎪 [COORDINATOR] Agent coordinateur initialisé")
        self.add_behaviour(self.CoordinateBehaviour())
 