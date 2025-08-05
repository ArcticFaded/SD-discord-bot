#!/usr/bin/env python3
"""
Startup script for SD Discord Bot with Temporal
Runs Temporal dev server, worker, and Discord bot without Docker
"""

import asyncio
import subprocess
import sys
import time
import signal
import os
from pathlib import Path

class BotManager:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_temporal_dev_server(self):
        """Start Temporal development server (requires temporal CLI installed)"""
        print("🚀 Starting Temporal development server...")
        
        # Check if temporal CLI is installed
        try:
            subprocess.run(["temporal", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Temporal CLI not found. Please install it first:")
            print("   Mac: brew install temporal")
            print("   Linux: curl -sSf https://temporal.download/cli.sh | sh")
            print("   Or visit: https://docs.temporal.io/cli#install")
            return None
        
        # Start temporal dev server
        process = subprocess.Popen(
            ["temporal", "server", "start-dev", "--ui-port", "8080", "--db-filename", "temporal.db"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.processes.append(("Temporal Server", process))
        
        # Wait for server to be ready
        time.sleep(5)
        print("✅ Temporal server started (UI at http://localhost:8080)")
        return process
    
    def start_temporal_worker(self):
        """Start the Temporal worker"""
        print("🚀 Starting Temporal worker...")
        
        process = subprocess.Popen(
            [sys.executable, "temporal_worker.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.processes.append(("Temporal Worker", process))
        
        # Wait for worker to connect
        time.sleep(3)
        print("✅ Temporal worker started")
        return process
    
    def start_discord_bot(self):
        """Start the Discord bot"""
        print("🚀 Starting Discord bot...")
        
        process = subprocess.Popen(
            [sys.executable, "run.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.processes.append(("Discord Bot", process))
        
        print("✅ Discord bot started")
        return process
    
    def monitor_processes(self):
        """Monitor all processes and restart if needed"""
        while self.running:
            for name, process in self.processes:
                if process.poll() is not None:
                    print(f"⚠️ {name} crashed with code {process.poll()}")
                    
                    # Print last output for debugging
                    stdout, stderr = process.communicate()
                    if stdout:
                        print(f"Last stdout from {name}:", stdout[-500:] if len(stdout) > 500 else stdout)
                    if stderr:
                        print(f"Last stderr from {name}:", stderr[-500:] if len(stderr) > 500 else stderr)
                    
                    # Restart the process
                    print(f"🔄 Restarting {name}...")
                    if "Temporal Server" in name:
                        self.start_temporal_dev_server()
                    elif "Temporal Worker" in name:
                        self.start_temporal_worker()
                    elif "Discord Bot" in name:
                        self.start_discord_bot()
            
            time.sleep(5)
    
    def cleanup(self, signum=None, frame=None):
        """Clean shutdown of all processes"""
        print("\n🛑 Shutting down all services...")
        self.running = False
        
        for name, process in self.processes:
            if process.poll() is None:
                print(f"  Stopping {name}...")
                process.terminate()
                
                # Give it time to shut down gracefully
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    process.kill()
        
        print("✅ All services stopped")
        sys.exit(0)
    
    async def run(self):
        """Main run loop"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        print("=" * 50)
        print("SD Discord Bot with Temporal Workflow Engine")
        print("=" * 50)
        
        # Start services in order
        temporal_process = self.start_temporal_dev_server()
        if not temporal_process:
            print("❌ Failed to start Temporal server")
            return
        
        worker_process = self.start_temporal_worker()
        if not worker_process:
            print("❌ Failed to start Temporal worker")
            self.cleanup()
            return
        
        bot_process = self.start_discord_bot()
        if not bot_process:
            print("❌ Failed to start Discord bot")
            self.cleanup()
            return
        
        print("\n" + "=" * 50)
        print("✅ All services running!")
        print("   - Temporal UI: http://localhost:8080")
        print("   - Discord bot: Active")
        print("   - Press Ctrl+C to stop all services")
        print("=" * 50 + "\n")
        
        # Monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            self.cleanup()

async def main():
    manager = BotManager()
    await manager.run()

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Check required files exist
    required_files = ["config.json", "temporal_worker.py", "run.py"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file not found: {file}")
            sys.exit(1)
    
    # Run the manager
    asyncio.run(main())