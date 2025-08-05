#!/usr/bin/env python3
"""
Startup script for DiffSynth backend server
"""

import uvicorn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    print("🚀 Starting DiffSynth Backend Server")
    print("=" * 50)
    print("Server will be available at: http://localhost:7860")
    print("API docs at: http://localhost:7860/docs")
    print("=" * 50)
    
    uvicorn.run(
        "diffsynth_server:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
        log_level="info"
    )