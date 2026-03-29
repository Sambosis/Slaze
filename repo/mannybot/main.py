#!/usr/bin/env python3
"""
Main entry point for the Mandelbrot Set Viewer application.
Initializes the PyQt5 application, creates the main window, and starts the event loop.
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from ui.main_window import MainWindow

def main() -> None:
    """
    Main function to initialize and run the Mandelbrot Set Viewer application.

    This function:
    1. Creates a QApplication instance
    2. Sets application properties and styles
    3. Creates the main window
    4. Shows the main window
    5. Starts the application event loop
    """
    # Create the application instance
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Mandelbrot Set Viewer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Mandelbrot Explorer")
    app.setStyle("Fusion")  # Use a modern Qt style

    # Set application icon if available
    try:
        icon = QIcon("assets/icon.png")
        app.setWindowIcon(icon)
    except FileNotFoundError:
        # Silently ignore if icon not found
        pass

    # Enable high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create the main window
    try:
        window = MainWindow()
    except Exception as e:
        print(f"Error creating main window: {e}", file=sys.stderr)
        sys.exit(1)

    # Show the main window
    window.show()

    # Start the event loop
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        # Allow clean exit on Ctrl+C
        sys.exit(0)

if __name__ == "__main__":
    main()
