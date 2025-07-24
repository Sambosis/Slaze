class IDEManager {
    constructor() {
        this.socket = io();
        this.initializeEventListeners();
        this.loadFileTree();
        this.loadMessages();
    }

    initializeEventListeners() {
        this.socket.on('update', (data) => {
            this.updateMessages(data);
        });
    }

    renderFileTree(files, container) {
        const ul = document.createElement('ul');
        files.forEach(file => {
            const li = document.createElement('li');
            li.textContent = file.name;
            
            if (file.type === 'directory') {
                li.classList.add('directory');
                li.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.toggleDirectory(li);
                });
                
                if (file.children) {
                    this.renderFileTree(file.children, li);
                }
            } else {
                li.classList.add('file');
                li.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.loadFileContent(file.path);
                });
            }
            ul.appendChild(li);
        });
        container.appendChild(ul);
    }

    toggleDirectory(directoryElement) {
        const childUl = directoryElement.querySelector('ul');
        if (childUl) {
            childUl.style.display = childUl.style.display === 'none' ? 'block' : 'none';
        }
    }

    async loadFileContent(filePath) {
        try {
            const response = await fetch(`/api/files/content?path=${filePath}`);
            const data = await response.json();
            const editor = document.getElementById('editor');
            editor.textContent = data.content;
            Prism.highlightElement(editor);
        } catch (error) {
            console.error('Error loading file content:', error);
        }
    }

    async loadFileTree() {
        try {
            const response = await fetch('/api/files');
            const data = await response.json();
            const fileTreeContainer = document.getElementById('file-tree');
            if (fileTreeContainer) {
                this.renderFileTree(data, fileTreeContainer);
            }
        } catch (error) {
            console.error('Error loading file tree:', error);
        }
    }

    updateMessages(data) {
        this.updateUserMessages(data.user);
        this.updateAssistantMessages(data.assistant);
    }

    updateUserMessages(userMessages) {
        const userMessagesDiv = document.getElementById('user-messages');
        if (!userMessagesDiv) return;

        userMessagesDiv.innerHTML = '';
        
        if (userMessages.length === 0) {
            userMessagesDiv.innerHTML = '<div class="empty-state">No user messages yet</div>';
        } else {
            userMessages.forEach(msg => {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message user-message';
                messageDiv.innerHTML = marked.parse(msg);
                userMessagesDiv.appendChild(messageDiv);
            });
        }
    }

    updateAssistantMessages(assistantMessages) {
        const assistantMessagesDiv = document.getElementById('assistant-messages');
        if (!assistantMessagesDiv) return;

        assistantMessagesDiv.innerHTML = '';
        
        if (assistantMessages.length === 0) {
            assistantMessagesDiv.innerHTML = '<div class="empty-state">No assistant messages yet</div>';
        } else {
            assistantMessages.forEach(msg => {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant-message';
                messageDiv.innerHTML = marked.parse(msg);
                assistantMessagesDiv.appendChild(messageDiv);
            });
        }
    }

    async loadMessages() {
        try {
            const response = await fetch('/messages');
            const data = await response.json();
            this.updateMessages(data);
        } catch (error) {
            console.error('Error loading messages:', error);
        }
    }
}

// Initialize the IDE when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new IDEManager();
});