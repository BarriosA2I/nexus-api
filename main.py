"""
Nexus API - Root Entry Point for Render Deployment
Re-exports the FastAPI app from backend.app.main
"""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and re-export the app
from backend.app.main import app

# This allows: uvicorn main:app
__all__ = ["app"]
