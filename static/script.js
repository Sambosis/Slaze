const socket = io();

function renderFileTree(files, container) {
    const ul = document.createElement('ul');
    files.forEach(file => {
        const li = document.createElement('li');
        li.textContent = file.name;
        if (file.type === 'directory') {
            li.classList.add('directory');
            li.addEventListener('click', (e) => {
                e.stopPropagation();
                const childUl = li.querySelector('ul');
                if (childUl) {
                    childUl.style.display = childUl.style.display === 'none' ? 'block' : 'none';
                }
            });
            if (file.children) {
                renderFileTree(file.children, li);
            }
        } else {
            li.classList.add('file');
            li.addEventListener('click', (e) => {
                e.stopPropagation();
                fetch(`/api/files/content?path=${file.path}`)
                    .then(response => response.json())
                    .then(data => {
                        const editor = document.getElementById('editor');
                        editor.textContent = data.content;
                        Prism.highlightElement(editor);
                    });
            });
        }
        ul.appendChild(li);
    });
    container.appendChild(ul);
}

fetch('/api/files')
    .then(response => response.json())
    .then(data => {
        const fileTreeContainer = document.getElementById('file-tree');
        renderFileTree(data, fileTreeContainer);
    });

socket.on('update', function(data) {
    const userMessagesDiv = document.getElementById('user-messages');
    userMessagesDiv.innerHTML = '';
    if (data.user.length === 0) {
        userMessagesDiv.innerHTML = '<div class="empty-state">No user messages yet</div>';
    } else {
        data.user.forEach(function(msg) {
            userMessagesDiv.innerHTML += '<div class="message user-message">' + marked.parse(msg) + '</div>';
        });
    }

    const assistantMessagesDiv = document.getElementById('assistant-messages');
    assistantMessagesDiv.innerHTML = '';
    if (data.assistant.length === 0) {
        assistantMessagesDiv.innerHTML = '<div class="empty-state">No assistant messages yet</div>';
    } else {
        data.assistant.forEach(function(msg) {
            assistantMessagesDiv.innerHTML += '<div class="message assistant-message">' + marked.parse(msg) + '</div>';
        });
    }
});

fetch('/messages')
    .then(response => response.json())
    .then(data => {
        socket.on('update', function(data) {
            const userMessagesDiv = document.getElementById('user-messages');
            userMessagesDiv.innerHTML = '';
            if (data.user.length === 0) {
                userMessagesDiv.innerHTML = '<div class="empty-state">No user messages yet</div>';
            } else {
                data.user.forEach(function(msg) {
                    userMessagesDiv.innerHTML += '<div class="message user-message">' + marked.parse(msg) + '</div>';
                });
            }

            const assistantMessagesDiv = document.getElementById('assistant-messages');
            assistantMessagesDiv.innerHTML = '';
            if (data.assistant.length === 0) {
                assistantMessagesDiv.innerHTML = '<div class="empty-state">No assistant messages yet</div>';
            } else {
                data.assistant.forEach(function(msg) {
                    assistantMessagesDiv.innerHTML += '<div class="message assistant-message">' + marked.parse(msg) + '</div>';
                });
            }
        });
    });
