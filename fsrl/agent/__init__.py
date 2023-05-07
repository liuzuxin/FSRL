"""The default MLP agent package."""

from fsrl.agent.base_agent import BaseAgent, OffpolicyAgent, OnpolicyAgent
from fsrl.agent.cpo_agent import CPOAgent
from fsrl.agent.cvpo_agent import CVPOAgent
from fsrl.agent.ddpg_lag_agent import DDPGLagAgent
from fsrl.agent.focops_agent import FOCOPSAgent
from fsrl.agent.ppo_lag_agent import PPOLagAgent
from fsrl.agent.sac_lag_agent import SACLagAgent
from fsrl.agent.trpo_lag_agent import TRPOLagAgent

__all__ = [
    "BaseAgent", "OffpolicyAgent", "OnpolicyAgent", "CVPOAgent", "SACLagAgent",
    "DDPGLagAgent", "PPOLagAgent", "TRPOLagAgent", "FOCOPSAgent", "CPOAgent"
]
