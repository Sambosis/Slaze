# Dockerized "Hello, World!" GUI Application (Tkinter)

## Description

This project demonstrates a simple "Hello, World!" GUI application written in Python using Tkinter, containerized with Docker, and configured for X11 forwarding. This allows the GUI to be displayed on your host operating system (Windows, Linux, macOS) even though the application is running inside a Docker container.  This provides a minimal, self-contained example of running graphical applications within Docker.

## Files

*   **`build_docker.py`:** Python script to build the Docker image and run the container.  It automatically handles X11 forwarding based on your host operating system (Windows, Linux, or macOS).
*   **`Dockerfile`:** Defines the Docker image.  It installs the necessary packages (XFCE, Python 3, Tkinter, Xvfb, and fluxbox) and configures the environment for running the GUI application.
*   **`entrypoint.sh`:** Bash script that is executed when the container starts. It performs the following tasks:
    *   Starts Xvfb (a virtual X server) on display `:99`.
    *   Starts fluxbox (a lightweight window manager) in the background.
    *   Runs the Python GUI application (`hello.py`).
    *   Includes error handling to ensure that all required components are available and started correctly.
    *     Uses `wait` to keep the container running.
*   **`hello.py`:** The Python GUI application itself.  It creates a simple window displaying "Hello, World!".

## Prerequisites

1.  **Docker:** You must have Docker installed and running on your system.  Use the following official Docker installation instructions for your operating system:
    *   **Windows:** [https://docs.docker.com/desktop/windows/install/](https://docs.docker.com/desktop/windows/install/) (Docker Desktop with WSL2 is strongly recommended)
    *   **Linux:** [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/) (Installation steps vary depending on your Linux distribution)
    *   **macOS:** [https://docs.docker.com/desktop/mac/install/](https://docs.docker.com/desktop/mac/install/)

2.  **X11 Server:** An X11 server must be running on your *host* machine to display the GUI from the Docker container. The specific X11 setup depends on your OS:
    *   **Windows:** VcXsrv is recommended. Download and install it from: [https://sourceforge.net/projects/vcxsrv/](https://sourceforge.net/projects/vcxsrv/)
    *   **Linux:** Most Linux distributions include Xorg (an X11 server) by default. If you have a graphical environment, it is most likely already running.
    *   **macOS:** XQuartz is required. Download and install it from: [https://www.xquartz.org/](https://www.xquartz.org/)

## Installation

1.  **Clone the repository:**