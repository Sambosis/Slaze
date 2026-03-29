import sys
import tkinter as tk
from typing import Optional

from src.app import DrumMachineApp


def main() -> None:
    root = tk.Tk()
    root.title("Drum Machine")
    app: Optional[DrumMachineApp] = None

    try:
        app = DrumMachineApp(root)
    except Exception:
        root.destroy()
        raise

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Closing drum machine...", file=sys.stderr)
    finally:
        if app is not None:
            shutdown = getattr(app, "on_closing", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    print(f"Error during shutdown: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()