from .i3d import I3D
from .motion_extractor import MotionExtractor
from .clip import CLIP
from .dino import DINO
from .laion import LAION
try:
    from .musiq import MUSIQ
except ImportError:
    MUSIQ = None  # pyiqa not installed (requires Python <3.10 due to llvmlite)
from .raft import RAFT
from .amt import AMT_S
