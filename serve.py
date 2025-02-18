# serve.py
from waitress import serve
from utils.agent_display_web_with_prompt import create_app  # Import create_app

if __name__ == '__main__':
    app = create_app()  # Create the app instance here
    app.debug = True  # Enable Flask's debug mode!  VERY IMPORTANT
    serve(app, host='0.0.0.0', port=5001, threads=4)  # threads is optional