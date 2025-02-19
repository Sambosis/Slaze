import asyncio
from utils.agent_display_web_with_prompt import create_app

# Create new event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Create app with the loop
app = create_app(loop)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001)
