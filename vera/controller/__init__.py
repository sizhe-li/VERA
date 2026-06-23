"""vera robot-side controller — chunk player that drives a robot from the policy server.

Runs ON the robot host (nora / DROID FR3), consumes ``WebsocketClientPolicy``. See
docs/CONTROLLER_SPEC.md. Hardware-free testing via the replay backend.
"""
from .robot_iface import RobotBackend, ReplayBackend
from .obs_builder import ObsBuilder
from .action_player import ActionPlayer
from .controller import VeraController

__all__ = ["RobotBackend", "ReplayBackend", "ObsBuilder", "ActionPlayer", "VeraController"]
