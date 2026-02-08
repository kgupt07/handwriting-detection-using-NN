"""Launch the handwriting recognition app. Run from project root: python run_app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.handwriting_app import main

if __name__ == "__main__":
    main()
