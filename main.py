"""
Nexus API - Root Entry Point for Render Deployment
Re-exports the FastAPI app from backend.app.main
"""
import sys
from pathlib import Path

# Add both project root AND backend to Python path
# This ensures 'from app.xxx' imports work when running from root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Import and re-export the app
from backend.app.main import app

# This allows: uvicorn main:app
__all__ = ["app"]
