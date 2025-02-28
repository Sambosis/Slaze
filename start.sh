#!/bin/bash

# Set the USER environment variable if not set
export USER=${USER:-myuser}

# Make sure we have the necessary directories
mkdir -p ~/.vnc
# Add this near the beginning of start.sh, before starting VNC server
# Clean up any stale VNC lock files
rm -f /tmp/.X1-lock
rm -f /home/myuser/.vnc/*.pid
rm -f /tmp/.X11-unix/X1
# Start VNC Server with explicitly defined font path
vncserver :1 -geometry 1280x800 -depth 24 -fp /usr/share/fonts/X11/misc/,/usr/share/fonts/X11/Type1/,/usr/share/fonts/X11/75dpi/,/usr/share/fonts/X11/100dpi/ || { echo "Failed to start VNC server" >&2; exit 1; }

# Give VNC server some time to initialize
sleep 5

# Set the DISPLAY environment variable
export DISPLAY=:0
# export DISPLAY=host.docker.internal:0.0

# Start XFCE Session
startxfce4 &> /dev/null &

# Give XFCE a little time to settle
sleep 2

# Set up development environment
# Create a Python virtual environment if it doesn't already exist
if [ ! -d "/home/myuser/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv /home/myuser/venv
    /home/myuser/venv/bin/pip install --upgrade pip
fi

# For development convenience, add desktop shortcuts
mkdir -p /home/myuser/Desktop
# mkdir -p /home/myuser/apps

# Create a terminal shortcut
cat > /home/myuser/Desktop/Terminal.desktop << 'EOL'
[Desktop Entry]
Version=1.0
Type=Application
Name=Terminal
Comment=Start a terminal
Exec=exo-open --launch TerminalEmulator
Icon=utilities-terminal
Path=
Terminal=false
StartupNotify=false
EOL

# Create a shortcut for Geany (code editor)
cat > /home/myuser/Desktop/Geany.desktop << 'EOL'
[Desktop Entry]
Version=1.0
Type=Application
Name=Geany Editor
Comment=Open Geany text editor
Exec=geany
Icon=geany
Path=
Terminal=false
StartupNotify=false
EOL
# Save display setting to profile for all future bash sessions
ENV DISPLAY=host.docker.internal:0.0

# echo "export DISPLAY=host.docker.internal:0.0" >> /home/myuser/.profile
# echo "export DISPLAY=host.docker.internal:0.0" >> /home/myuser/.bashrc
echo "export DISPLAY=:0" >> /home/myuser/.profile
echo "export DISPLAY=:0" >> /home/myuser/.bashrc
chmod +x /home/myuser/Desktop/*.desktop

# Setup bash aliases for development
cat > /home/myuser/.bash_aliases << 'EOL'
# Python aliases
alias python=python3
alias pip=pip3
alias ipython=ipython3
alias activate="source /home/myuser/venv/bin/activate"

# Development shortcuts
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Print welcome message
echo "Python development environment ready."
echo "Your local repo/ directory is mounted at /home/myuser/apps"
EOL

# Optional: Start Jupyter notebook server in background (uncomment if needed)
nohup jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root &> /home/myuser/jupyter.log &

echo "Development environment setup complete!"

# Keep the script running
wait