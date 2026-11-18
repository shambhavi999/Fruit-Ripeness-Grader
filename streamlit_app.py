"""Entrypoint for Streamlit Community Cloud.
Dispatches directly to app.py so both 'app.py' and 'streamlit_app.py' work seamlessly as the main file path.
"""
from pathlib import Path
import runpy
import sys

_app_file = Path(__file__).resolve().parent / "app.py"

# Execute the primary Streamlit application
runpy.run_path(str(_app_file), run_name="__main__")
