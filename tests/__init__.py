import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "classfactory"))
