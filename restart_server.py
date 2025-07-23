#!/usr/bin/env python3
"""
Server restart script to help resolve web component conflicts and server errors.
"""

import sys
import socket
import subprocess
from pathlib import Path

def check_port(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False  # Port is free
        except OSError:
            return True  # Port is in use

def find_free_port(start_port=5002, max_attempts=10):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not check_port(port):
            return port
    return None

def clear_cache():
    """Clear application cache."""
    cache_dir = Path("cache")
    if cache_dir.exists():
        for file in cache_dir.glob("*"):
            try:
                if file.is_file():
                    file.unlink()
                    print(f"Cleared cache file: {file}")
            except Exception as e:
                print(f"Could not clear {file}: {e}")

def main():
    print("🔧 Slazy Agent Server Restart Tool")
    print("=" * 40)
    
    # Clear cache
    print("\n1. Clearing application cache...")
    clear_cache()
    
    # Find free port
    print("\n2. Finding available port...")
    port = find_free_port()
    if not port:
        print("❌ No free ports found in range 5002-5012")
        return 1
    
    print(f"✅ Using port {port}")
    
    # Start server
    print(f"\n3. Starting server on port {port}...")
    print("💡 Tips to avoid web component conflicts:")
    print("   - Use incognito/private browsing mode")
    print("   - Disable browser extensions")
    print("   - Clear browser cache (Ctrl+Shift+Delete)")
    print("   - Try a different browser if issues persist")
    print()
    
    try:
        cmd = [sys.executable, "run.py", "web", "--port", str(port)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return 1
    except FileNotFoundError:
        print("❌ Could not find run.py. Make sure you're in the correct directory.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
