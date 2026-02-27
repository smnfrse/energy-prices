import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment and return project root."""
    current = Path.cwd()
    
    if current.name == 'notebooks':
        PROJ_ROOT = current.parent
    else:
        PROJ_ROOT = current
    
    # Change directory
    os.chdir(PROJ_ROOT)
    
    # Add to Python path
    sys.path.insert(0, str(PROJ_ROOT))
    
    print(f"Setup complete. Root: {PROJ_ROOT}")
    return PROJ_ROOT

# Run setup when imported
PROJ_ROOT = setup_environment()