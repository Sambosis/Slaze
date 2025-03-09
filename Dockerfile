FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV USER=myuser
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set default display for all sessions
ENV DISPLAY=:0
# In your Dockerfile, add:
ENV DISPLAY=host.docker.internal:0
# Install development packages, XFCE, Python tools, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    xfce4 \
    xfce4-goodies \
    tightvncserver \
    python3 \
    python3-pip \
    python3-tk \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    curl \
    wget \
    sudo \
    xfonts-base \
    xfonts-75dpi \
    xfonts-100dpi \
    xfonts-scalable \
    fonts-liberation \
    xauth \
    vim \
    nano \
    geany \
    ipython3 \
    dbus \
    dbus-x11 \
    openssh-client && \
    # Create a user 'myuser' with password 'mypassword' and sudo privileges
    useradd -m -s /bin/bash myuser && \
    echo "myuser:mypassword" | chpasswd && \
    adduser myuser sudo && \
    echo "myuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/myuser && \
    # Set up VNC password for 'myuser'
    mkdir -p /home/myuser/.vnc && \
    echo 'mypassword' | vncpasswd -f > /home/myuser/.vnc/passwd && \
    chmod 600 /home/myuser/.vnc/passwd && \
    chown -R myuser:myuser /home/myuser/.vnc && \
    # Create .Xauthority file
    touch /home/myuser/.Xauthority && \
    chown myuser:myuser /home/myuser/.Xauthority && \
    # Install common Python packages
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    notebook \
    matplotlib \
    pygame \
    torch   \
    numpy \
    pandas \
    pytest \
    pylint \
    black \
    flake8 \
    autopep8 \
    requests && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*
# Install Node.js and npm (using NodeSource's apt repo for a modern release)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Optional: Install global CLI tools to support various web frameworks and backends
RUN npm install -g \
    @angular/cli \       # For Angular projects
    create-react-app \   # For Create React App (optional)
    create-next-app \    # For Next.js scaffolding (optional)
    @vue/cli \           # For Vue CLI
    create-nuxt-app \    # For Nuxt.js (optional)
    express-generator \  # For Express scaffolding (optional)
    @nestjs/cli \        # For NestJS scaffolding
    yarn                 # If you prefer Yarn over npm
# Create app directory that will be mounted
RUN mkdir -p /home/myuser/apps && \
    chown -R myuser:myuser /home/myuser/apps

# Copy the start script
COPY start.sh /home/myuser/
RUN chmod +x /home/myuser/start.sh && \
    chown myuser:myuser /home/myuser/start.sh

# Set the working directory to the app directory
WORKDIR /home/myuser/apps

# Switch to the 'myuser' user
USER myuser

# Expose ports for VNC and potentially Jupyter Notebook
EXPOSE 5901 8888

# Use the start script as the default command
CMD ["/home/myuser/start.sh"]