// file_browser.js

// Global socket variable
let socket = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeSocket();
    refreshFileTree();
    document.getElementById('download-zip-btn').addEventListener('click', downloadZip);
    document.getElementById('download-file-btn').addEventListener('click', downloadCurrentFile);
});

function initializeSocket() {
    if (socket && socket.connected) {
        console.log('🔄 Socket already connected in file browser');
        return;
    }
    
    console.log('🚀 Initializing socket in file browser');
    socket = io({
        autoConnect: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000
    });

    socket.on('connect', function() {
        console.log('✅ File browser socket connected with ID:', socket.id);
        setupSocketListeners();
    });
    
    socket.on('disconnect', function(reason) {
        console.log('❌ File browser socket disconnected:', reason);
    });

    socket.on('connect_error', function(error) {
        console.error('❌ File browser socket connection error:', error);
    });
}

let currentFilePath = null;

function refreshFileTree() {
    fetch('/api/file-tree')
        .then(response => response.json())
        .then(tree => {
            const rootContents = document.getElementById('root-contents');
            rootContents.innerHTML = '';
            renderTree(tree, rootContents);
        })
        .catch(error => console.error('Error loading file tree:', error));
}

function renderTree(nodes, container) {
    nodes.forEach(node => {
        const div = document.createElement('div');
        div.className = node.type === 'directory' ? 'folder-item' : 'file-item';
        div.innerHTML = `<i class="fas fa-${node.type === 'directory' ? 'folder' : 'file'}"></i> ${node.name}`;
        div.dataset.path = node.path;
        
        if (node.type === 'directory') {
            const contents = document.createElement('div');
            contents.className = 'folder-contents';
            contents.style.display = 'none';
            div.addEventListener('click', (e) => {
                e.stopPropagation();
                contents.style.display = contents.style.display === 'none' ? 'block' : 'none';
            });
            renderTree(node.children, contents);
            div.appendChild(contents);
        } else {
            div.addEventListener('click', (e) => {
                e.stopPropagation();
                openFile(node.path);
            });
        }
        
        container.appendChild(div);
    });
}

function openFile(path) {
    fetch(`/api/file-content?path=${encodeURIComponent(path)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load file');
            }
            return response.text();
        })
        .then(content => {
            const editorContent = document.getElementById('editor-content');
            editorContent.innerHTML = `<pre><code>${escapeHtml(content)}</code></pre>`;
            // Update tabs or something, for now simple
            currentFilePath = path;
            document.getElementById('download-file-btn').style.display = 'inline-block';
        })
        .catch(error => {
            document.getElementById('editor-content').innerHTML = `<div class="error">Error loading file: ${error.message}</div>`;
        });
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

function downloadZip() {
    window.location.href = '/api/download-zip';
}

function downloadCurrentFile() {
    if (currentFilePath) {
        window.location.href = `/api/download?path=${encodeURIComponent(currentFilePath)}`;
    }
}

// Handle socket updates for console
function setupSocketListeners() {
    if (!socket) {
        console.error('❌ Socket not initialized in file browser');
        return;
    }
    
    socket.on('update', function(data) {
        console.log('📨 File browser received update:', data);
        const consoleMessagesDiv = document.getElementById('console-messages');
        if (consoleMessagesDiv) {
            // Clear existing
            consoleMessagesDiv.innerHTML = '';
            // Add new messages
            if (data.assistant && data.assistant.length > 0) {
                data.assistant.forEach(msg => {
                    addConsoleMessage(msg, 'assistant');
                });
            } else {
                consoleMessagesDiv.innerHTML = '<div class="empty-state">Console output will appear here</div>';
            }
        }
    });
}

function addConsoleMessage(message, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `console-message ${type}`;
    messageDiv.innerHTML = `<pre>${escapeHtml(message)}</pre>`;
    const consoleMessagesDiv = document.getElementById('console-messages');
    consoleMessagesDiv.appendChild(messageDiv);
    consoleMessagesDiv.scrollTop = consoleMessagesDiv.scrollHeight;
}
