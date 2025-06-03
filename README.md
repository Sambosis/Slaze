# ✨ Slazy - Interactive Agent Interface 🤖

## 🚀 Project Purpose
Slazy is your go-to interactive agent interface, providing a web-based playground 🏖️ for seamless communication with AI agents. Experience real-time message updates 💬, tool execution results 🛠️, and effortless project file management 🗂️, all wrapped in a clean, modern interface.

## ✨ Key Features
- **🖥️ Web Interface**: Modern, responsive design powered by TailwindCSS 🎨
- **⚡ Real-time Updates**: Socket.IO based communication ensures you're always in the loop 🔄
- **🗂️ Message Categories**: Dedicated panels for Assistant 🧞‍♀️, Tool 🧰, and User 🤡 messages
- **💾 File Management**: Download generated files with ease ⬇️
- **🛑 Interrupt Functionality**: Halt agent operations on demand ⛔
- **🧰 Tool Integration**: Supports a variety of tools, including code writing ✍️ and project setup ⚙️
- **⚙️ Environment Management**: Manages environment variables and dependencies like a pro 💼

## 🎯 Use Cases
- **Rapid Prototyping**: Quickly test and iterate on AI agent designs 🧪.
- **Educational Tool**: Great for learning about AI agents and how they interact with tools 📚.
- **Project Management**: Automate tasks and manage project files efficiently 📈.
- **Custom Solutions**: Build tailored AI solutions for specific needs 🛠️.

## ⚠️ Prerequisites
- Python 3.8+ 🐍
- Modern web browser (Chrome/Firefox/Safari recommended) 🌐
- Required API keys (set in `.env` file):
  - `ANTHROPIC_API_KEY` 🔑

## 🛠️ Setup Instructions
1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/slazy.git
    cd slazy
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On Unix/MacOS:
    source .venv/bin/activate
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create a `.env` file in the root directory with:
    ```
    ANTHROPIC_API_KEY=your_api_key_here
    ```

## 🏃 Running the Application

1. **Start the application**:
    ```sh
    python run.py
    ```
    This will:
    - 🚀 Start the Flask server
    - 🌐 Open your default web browser to http://127.0.0.1:5001/select_prompt
    - 🤖 Initialize the agent interface

2. **Using the Interface**:
    - The interface consists of three main panels:
        - Assistant Messages (top left) 🧞‍♀️
        - Tool Results (top right) 🧰
        - User Messages (bottom) 🤡
    - Type messages in the input field at the bottom ⌨️
    - Use the "Send" button or press Enter to send messages ✉️
    - Use the "Interrupt Agent" button to stop current operations 🛑
    - Download generated files using the download section 💾

## 📝 Example Usage

1. **Selecting a Prompt**:
   - Upon starting the application, you'll be directed to the `/select_prompt` route.
   - Here, you can either select a pre-existing prompt from the dropdown menu or create a new one.
   - If creating a new prompt, provide a filename and the prompt text in the respective fields.
   - Once a prompt is selected or created, the agent will begin processing the task outlined in the prompt.

2. **Interacting with the Agent**:
   - Use the input field at the bottom of the main interface to send instructions or questions to the agent.
   - The agent's responses, along with any tool execution results, will be displayed in their respective panels in real-time.

3. **Downloading Project Files**:
   - To download the entire project as a ZIP archive, click the "Download Project ZIP" button located in the download links section.
   - This will package all project files into a single archive for easy access and sharing.

## 📂 Project Structure
```
slazy/
├── templates/
│   └── index.html          # Main web interface template 🖼️
├── utils/
│   ├── agent_display_web_with_prompt.py
│   ├── context_helpers.py
│   ├── file_logger.py
│   └── output_manager.py
├── tools/
│   └── [tool implementation files] 🛠️
├── main.py                 # Core agent logic 🧠
├── run.py                 # Application entry point 🚪
├── requirements.txt       # Project dependencies 📦
└── README.md
```

## 🐛 Error Handling
- The interface provides visual feedback for errors 🚨
- Check the console for detailed error messages ℹ️
- Logs are stored in the `logs` directory 📝

## 🛠️ Troubleshooting
1. **Server won't start**:
   - Check if port 5001 is available 🚫
   - Ensure all dependencies are installed ✅
   - Verify Python version compatibility 🐍

2. **Connection issues**:
   - Check if the server is running 🏃
   - Verify WebSocket connection 🔗
   - Check browser console for errors 🕵️

3. **API issues**:
   - Verify API key in `.env` file 🔑
   - Check API rate limits ⏳
   - Ensure internet connectivity 🌐

## 🤝 Contributing
1. Fork the repository 🍴
2. Create your feature branch 🌿
3. Commit your changes ✍️
4. Push to the branch 📤
5. Create a Pull Request 🤝

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments
- Uses OpenAI models for AI capabilities 🤖
- Built with Flask and Socket.IO 💡
- Styled with TailwindCSS 🎨
