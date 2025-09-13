#!/usr/bin/env python3
"""Start the AI Crypto Trader Dashboard."""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the dashboard server."""
    print("🚀 Starting AI Crypto Trader Dashboard...")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment.")
        print("   It's recommended to activate the virtual environment first:")
        print("   source venv/bin/activate")
        print()
    
    # Check if required files exist
    required_files = [
        'web_server.py',
        'ai_trader.py',
        'market_analyzer.py',
        'kraken_client.py',
        'risk_manager.py',
        'trading_executor.py',
        'performance_analyzer.py',
        'config.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all trading system files are present.")
        return 1
    
    # Check if frontend files exist
    frontend_files = [
        'frontend/dist/index.html'
    ]
    
    missing_frontend = []
    for file in frontend_files:
        if not Path(file).exists():
            missing_frontend.append(file)
    
    if missing_frontend:
        print("⚠️  Frontend files not found. Dashboard will show a basic interface.")
        print("   To build the full React frontend, run:")
        print("   cd frontend && npm install && npm run build")
        print()
    
    print("🌐 Dashboard will be available at:")
    print("   http://localhost:8000")
    print()
    print("📚 API documentation at:")
    print("   http://localhost:8000/docs")
    print()
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the web server
        subprocess.run([sys.executable, 'web_server.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Goodbye!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting dashboard: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())