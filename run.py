import subprocess
import webbrowser
import time

# Run the UV command
subprocess.Popen(["uv", "run", "main.py"])

# Wait a moment for the server to start
time.sleep(2)

# Open the browser
webbrowser.open("http://127.0.0.1:5000/select_prompt")
     