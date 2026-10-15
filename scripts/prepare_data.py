#!/usr/bin/env python3
"""Prepare canonical datasets used by FedCode."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedcode.preparation import main
if __name__ == "__main__":
    main()
