# reads in a .env file and sets the environment variables in the OS permanently
# usage: python envtoenv.py .env

import os
import sys
import subprocess

def set_env_variables_from_file(file_path, permanent=False):
    with open(file_path, "r") as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue
            
            # Handle lines with = in the value
            parts = line.strip().split("=", 1)
            if len(parts) != 2:
                continue
                
            key, value = parts
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Remove quotes if present
            
            # Set for current session
            os.environ[key] = value
            
            # Set permanently if requested
            if permanent:
                try:
                    # Use setx to set permanently for current user
                    subprocess.run(["setx", key, value], check=True, capture_output=True)
                    print(f"Permanently set {key}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to permanently set {key}: {e}")

if __name__ == "__main__":  
    if len(sys.argv) < 2:
        print("Usage: python envtoenv.py .env [--permanent]")
        print("  --permanent: Set variables permanently (requires restart for new processes)")
        sys.exit(1)

    env_file = sys.argv[1]
    permanent = "--permanent" in sys.argv
    
    set_env_variables_from_file(env_file, permanent)
    
    if permanent:
        print(f"Environment variables permanently set from {env_file}")
        print("Note: You may need to restart applications/terminal for changes to take effect")
    else:
        print(f"Environment variables set for current session from {env_file}")