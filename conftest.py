"""
FreshChain AI — Pytest configuration
Ensures workspace root is in sys.path during test execution.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
