# serve.py
from waitress import serve
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt, app  # Import your app
import main
if __name__ == '__main__':
    main.main()
    serve(app, host='0.0.0.0', port=5001)  # Or any port you like

    