from utils.agent_display_web_with_prompt import create_app

# This makes 'app' visible to Gunicorn:
app = create_app()

if __name__ == '__main__':
    # For local dev:
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001, threads=4)
