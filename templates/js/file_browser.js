(function() {
    // Check if FontAwesome is loaded, if not, load it.
    if (!document.querySelector('link[href*="font-awesome"]')) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css';
        document.head.appendChild(link);
    }

    let openTabs = [];
    let activeTab = null;
    let expandedFolders = new Set();

    async function loadFileTree() {
        try {
            const response = await fetch('/api/file-tree');
            const fileTree = await response.json();
            const rootContents = document.getElementById('root-contents');
            if(rootContents) {
                renderFileTree(fileTree, rootContents);
            }
        } catch (error) {
            console.error('Error loading file tree:', error);
        }
    }

    function refreshFileTree() { loadFileTree(); }
    window.refreshFileTree = refreshFileTree; // Make it globally accessible for the onclick

    function renderFileTree(items, container) {
        container.innerHTML = '';
        items.forEach(item => {
            if (item.type === 'directory') {
                const folderDiv = document.createElement('div');
                folderDiv.className = 'folder-item';
                folderDiv.dataset.path = item.path;

                const contentsDiv = document.createElement('div');
                contentsDiv.className = 'folder-contents';

                const wasExpanded = expandedFolders.has(item.path);
                if (wasExpanded) {
                    contentsDiv.classList.remove('hidden');
                    folderDiv.innerHTML = `<i class="fas fa-folder-open"></i> <span>${item.name}</span>`;
                    if (item.children && item.children.length > 0) {
                        renderFileTree(item.children, contentsDiv);
                    }
                } else {
                    contentsDiv.classList.add('hidden');
                    folderDiv.innerHTML = `<i class="fas fa-folder"></i> <span>${item.name}</span>`;
                }

                folderDiv.addEventListener('click', (e) => {
                    e.stopPropagation();
                    toggleFolder(folderDiv, contentsDiv, item.children);
                });

                container.appendChild(folderDiv);
                container.appendChild(contentsDiv);
            } else {
                const fileDiv = document.createElement('div');
                fileDiv.className = 'file-item';
                fileDiv.dataset.path = item.path;

                const icon = getFileIcon(item.name);
                fileDiv.innerHTML = `<i class="${icon.class}"></i> <span>${item.name}</span>`;

                fileDiv.addEventListener('click', () => openFile(item.path, item.name));
                container.appendChild(fileDiv);
            }
        });
    }

    function toggleFolder(folderDiv, contentsDiv, children) {
        const icon = folderDiv.querySelector('i');
        const isOpen = !contentsDiv.classList.contains('hidden');
        const folderPath = folderDiv.dataset.path;

        if (isOpen) {
            contentsDiv.classList.add('hidden');
            icon.className = 'fas fa-folder';
            expandedFolders.delete(folderPath);
        } else {
            contentsDiv.classList.remove('hidden');
            icon.className = 'fas fa-folder-open';
            expandedFolders.add(folderPath);
            if (children && children.length > 0 && contentsDiv.innerHTML === "") {
                renderFileTree(children, contentsDiv);
            }
        }
    }

    function getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const icons = {
            'py': { class: 'fab fa-python' }, 'js': { class: 'fab fa-js-square' },
            'html': { class: 'fab fa-html5' }, 'css': { class: 'fab fa-css3-alt' },
            'json': { class: 'fas fa-code' }, 'md': { class: 'fab fa-markdown' },
            'txt': { class: 'fas fa-file-alt' }
        };
        return icons[ext] || { class: 'fas fa-file' };
    }

    async function openFile(filePath, fileName) {
        if (openTabs.find(tab => tab.path === filePath)) {
            switchToTab(openTabs.find(tab => tab.path === filePath));
            return;
        }

        const editorContent = document.getElementById('editor-content');
        if(editorContent) editorContent.innerHTML = `<div class="editor-welcome"><i class="fas fa-spinner fa-spin"></i><h3>Loading...</h3></div>`;

        try {
            const response = await fetch(`/api/file-content?path=${encodeURIComponent(filePath)}`);
            const content = await response.text();

            const tab = { path: filePath, name: fileName, content: content };
            openTabs.push(tab);
            renderTabs();
            switchToTab(tab);
        } catch (error) {
            console.error('Error opening file:', error);
            if(editorContent) editorContent.innerHTML = `<div class="editor-welcome"><h3>Error loading file</h3></div>`;
        }
    }

    function renderTabs() {
        const tabsContainer = document.getElementById('editor-tabs');
        if(!tabsContainer) return;
        tabsContainer.innerHTML = '';
        openTabs.forEach(tab => {
            const tabDiv = document.createElement('div');
            tabDiv.className = `editor-tab ${tab === activeTab ? 'active' : ''}`;
            tabDiv.innerHTML = `<span>${tab.name}</span><span class="close-btn" data-path="${tab.path}">&times;</span>`;

            tabDiv.addEventListener('click', (e) => {
                if (e.target.classList.contains('close-btn')) {
                    closeTab(e.target.dataset.path);
                } else {
                    switchToTab(tab);
                }
            });
            tabsContainer.appendChild(tabDiv);
        });
    }

    function switchToTab(tab) {
        activeTab = tab;
        renderTabs();

        const editorContent = document.getElementById('editor-content');
        if(!editorContent) return;

        const language = getLanguageFromFilename(tab.name);
        editorContent.innerHTML = `<pre><code class="language-${language}">${escapeHtml(tab.content)}</code></pre>`;
        Prism.highlightAll();

        const dlBtn = document.getElementById('download-file-btn');
        if (dlBtn) {
            dlBtn.style.display = 'inline-block';
            dlBtn.dataset.path = tab.path;
        }
    }

    function closeTab(filePath) {
        openTabs = openTabs.filter(t => t.path !== filePath);
        if (activeTab && activeTab.path === filePath) {
            activeTab = openTabs.length > 0 ? openTabs[openTabs.length - 1] : null;
        }
        if (activeTab) {
            switchToTab(activeTab);
        } else {
            const editorContent = document.getElementById('editor-content');
            if(editorContent) editorContent.innerHTML = `<div class="editor-welcome">...</div>`;
            renderTabs();
            const dlBtn = document.getElementById('download-file-btn');
            if (dlBtn) dlBtn.style.display = 'none';
        }
    }

    function getLanguageFromFilename(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        return { 'py': 'python', 'js': 'javascript', 'html': 'html', 'css': 'css', 'json': 'json', 'md': 'markdown', 'sh': 'bash' }[ext] || 'text';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function initializeResizer() {
        const resizer = document.getElementById('browser-sidebar-resizer');
        const container = document.querySelector('#main-content-area .vscode-container');
        if(!resizer || !container) return;

        let isResizing = false;
        resizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            document.body.style.cursor = 'col-resize';
            e.preventDefault();
        });
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            const rect = container.getBoundingClientRect();
            const newWidth = e.clientX - rect.left;
            if (newWidth > 150 && newWidth < rect.width - 150) {
                container.style.gridTemplateColumns = `${newWidth}px 4px 1fr`;
            }
        });
        document.addEventListener('mouseup', () => {
            isResizing = false;
            document.body.style.cursor = '';
        });
    }

    // --- Init ---
    loadFileTree();
    initializeResizer();

    document.getElementById('download-zip-btn')?.addEventListener('click', () => {
        window.location.href = '/api/download-zip';
    });
    document.getElementById('download-file-btn')?.addEventListener('click', () => {
        const path = document.getElementById('download-file-btn').dataset.path;
        if(path) window.location.href = `/api/download?path=${encodeURIComponent(path)}`;
    });

})();
