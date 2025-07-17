
Spry.Utils.addLoadListener(function() {

        var socket = io();
        var currentTab = 'user';
        var currentToolData = null;

        socket.on('connect', function() {
            console.log('Connected to SocketIO');
        });

        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(function(content) {
                content.classList.remove('active');
            });
            
            // Remove active class from all tab buttons
            document.querySelectorAll('.tab-button').forEach(function(button) {
                button.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName + '-content').classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
            
            currentTab = tabName;
        }

        function updateMessages(data) {
            // Update user messages
            var userMessagesDiv = document.getElementById('user-messages');
            userMessagesDiv.innerHTML = '';
            if (data.user.length === 0) {
                userMessagesDiv.innerHTML = '<div class="empty-state">No user messages yet</div>';
            } else {
                data.user.forEach(function(msg) {
                    userMessagesDiv.innerHTML += '<div class="message user-message">' + escapeHtml(msg) + '</div>';
                });
            }
            
            // Update assistant messages
            var assistantMessagesDiv = document.getElementById('assistant-messages');
            assistantMessagesDiv.innerHTML = '';
            if (data.assistant.length === 0) {
                assistantMessagesDiv.innerHTML = '<div class="empty-state">No assistant messages yet</div>';
            } else {
                data.assistant.forEach(function(msg) {
                    assistantMessagesDiv.innerHTML += '<div class="message assistant-message">' + parseMarkdown(msg) + '</div>';
                });
            }
            
            // Update tool messages
            var toolMessagesDiv = document.getElementById('tool-messages');
            toolMessagesDiv.innerHTML = '';
            if (data.tool.length === 0) {
                toolMessagesDiv.innerHTML = '<div class="empty-state">No tool results yet</div>';
            } else {
                data.tool.forEach(function(msg) {
                    // Directly use msg as it's expected to be HTML from tools
                    // The outer div structure is kept for consistent message styling
                    toolMessagesDiv.innerHTML += '<div class="message tool-message">' + msg + '</div>';
                });
            }
            
            // Update badge counts
            document.getElementById('user-count').textContent = data.user.length;
            document.getElementById('assistant-count').textContent = data.assistant.length;
            document.getElementById('tool-count').textContent = data.tool.length;
            
            // Scroll current tab to bottom
            var currentMessagesDiv = document.getElementById(currentTab + '-messages');
            if (currentMessagesDiv) {
                currentMessagesDiv.scrollTop = currentMessagesDiv.scrollHeight;
            }
            
            // Re-highlight code blocks
            if (typeof Prism !== 'undefined') {
                Prism.highlightAll();
            }
        }

        function escapeHtml(text) {
            // Convert non-string values to string first
            if (typeof text !== 'string') {
                if (text === null || text === undefined) {
                    text = '';
                } else if (typeof text === 'object') {
                    text = JSON.stringify(text);
                } else {
                    text = String(text);
                }
            }
            
            var map = {
                '&': '&',
                '<': '<',
                '>': '>',
                '"': '"',
                "'": '''
            };
            return text.replace(/[&<>"']/g, function(m) { return map[m]; });
        }

        function parseMarkdown(text) {
            // Convert non-string values to string first
            if (typeof text !== 'string') {
                if (text === null || text === undefined) {
                    text = '';
                } else if (typeof text === 'object') {
                    text = JSON.stringify(text);
                } else {
                    text = String(text);
                }
            }
            
            // Configure marked to support console language highlighting
            marked.setOptions({
                highlight: function(code, lang) {
                    if (lang && Prism.languages[lang]) {
                        return Prism.highlight(code, Prism.languages[lang], lang);
                    }
                    // For console blocks, use shell-session highlighting if available
                    if (lang === 'console' && Prism.languages['shell-session']) {
                        return Prism.highlight(code, Prism.languages['shell-session'], 'shell-session');
                    }
                    // Fall back to bash highlighting for console blocks
                    if (lang === 'console' && Prism.languages['bash']) {
                        return Prism.highlight(code, Prism.languages['bash'], 'bash');
                    }
                    return code;
                }
            });
            
            return marked.parse(text);
        }

        socket.on('update', function(data) {
            updateMessages(data);
        });

        socket.on('agent_prompt', function(data) {
            var promptArea = document.getElementById('agent_prompt_area');
            if (data.message) {
                promptArea.textContent = data.message;
                promptArea.style.display = 'block';
            } else {
                promptArea.textContent = '';
                promptArea.style.display = 'none';
            }
        });

        socket.on('agent_prompt_clear', function() {
            var promptArea = document.getElementById('agent_prompt_area');
            promptArea.textContent = '';
            promptArea.style.display = 'none';
        });

        socket.on('tool_prompt', function(data) {
            showToolPrompt(data);
        });

        socket.on('tool_prompt_clear', function() {
            hideToolPrompt();
        });

        function getToolIcon(toolName) {
            const icons = {
                'bash': '💻',
                'write_code': '📝',
                'edit': '✏️',
                'project_setup': '🏗️',
                'picture_generation': '🎨',
                'file_operations': '📁',
                'search': '🔍',
                'default': '🔧'
            };
            return icons[toolName] || icons.default;
        }

        function getToolDescription(toolName, schema) {
            if (schema && schema.description) {
                return schema.description;
            }
            
            const descriptions = {
                'bash': 'Execute shell commands in the terminal',
                'write_code': 'Create or modify code files',
                'edit': 'Edit existing files with specific changes',
                'project_setup': 'Set up project structure and dependencies',
                'picture_generation': 'Generate images using AI',
                'default': 'Review and modify the parameters before executing this tool.'
            };
            return descriptions[toolName] || descriptions.default;
        }

        function createFormField(param, info, value) {
            const group = document.createElement('div');
            group.className = 'form-group';
            
            const label = document.createElement('label');
            label.className = 'form-label';
            label.textContent = param;
            
            if (info.required) {
                const required = document.createElement('span');
                required.className = 'required';
                required.textContent = '*';
                label.appendChild(required);
            } else {
                const optional = document.createElement('span');
                optional.className = 'optional';
                optional.textContent = ' (optional)';
                label.appendChild(optional);
            }
            
            group.appendChild(label);
            
            let input;
            
            if (info.enum) {
                input = document.createElement('select');
                input.className = 'form-control';
                
                // Add empty option for optional fields
                if (!info.required) {
                    const emptyOption = document.createElement('option');
                    emptyOption.value = '';
                    emptyOption.textContent = '-- Select --';
                    input.appendChild(emptyOption);
                }
                
                info.enum.forEach(function(opt) {
                    const option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt;
                    if (opt == value) option.selected = true;
                    input.appendChild(option);
                });
            } else if (info.type === 'boolean') {
                const checkboxGroup = document.createElement('div');
                checkboxGroup.className = 'checkbox-group';
                
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = value === true || value === 'true';
                
                const checkboxLabel = document.createElement('label');
                checkboxLabel.textContent = 'Enable';
                
                checkboxGroup.appendChild(input);
                checkboxGroup.appendChild(checkboxLabel);
                group.appendChild(checkboxGroup);
            } else if (info.type === 'array') {
                const arrayContainer = document.createElement('div');
                arrayContainer.className = 'array-input';
                
                const tagsContainer = document.createElement('div');
                tagsContainer.className = 'array-tags';
                
                input = document.createElement('input');
                input.type = 'text';
                input.className = 'form-control';
                input.placeholder = 'Type and press Enter to add items';
                
                // Initialize with existing values
                let arrayValue = Array.isArray(value) ? value : (value ? [value] : []);
                
                // Create a unique ID for this array field
                const arrayId = 'array_' + Math.random().toString(36).substr(2, 9);
                
                function updateTags() {
                    tagsContainer.innerHTML = '';
                    arrayValue.forEach(function(item, index) {
                        const tag = document.createElement('span');
                        tag.className = 'array-tag';
                        tag.innerHTML = escapeHtml(item) + '<span class="remove" onclick="removeArrayItem_' + arrayId + '(' + index + ')">×</span>';
                        tagsContainer.appendChild(tag);
                    });
                }
                
                // Create a unique function for this array field
                window['removeArrayItem_' + arrayId] = function(index) {
                    arrayValue.splice(index, 1);
                    updateTags();
                };
                
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && this.value.trim()) {
                        e.preventDefault();
                        arrayValue.push(this.value.trim());
                        this.value = '';
                        updateTags();
                    }
                });
                
                // Store array value accessor
                input.getArrayValue = function() { return arrayValue; };
                
                updateTags();
                arrayContainer.appendChild(tagsContainer);
                arrayContainer.appendChild(input);
                group.appendChild(arrayContainer);
            } else if (info.type === 'integer') {
                input = document.createElement('input');
                input.type = 'number';
                input.className = 'form-control';
                input.value = value || '';
                input.step = '1';
            } else {
                input = document.createElement('textarea');
                input.className = 'form-control';
                input.value = value || '';
                input.rows = info.type === 'string' && (value || '').length > 100 ? 4 : 2;
            }
            
            if (input) {
                input.name = param;
                if (info.type !== 'boolean' && info.type !== 'array') {
                    group.appendChild(input);
                }
            }
            
            // Add help text if available
            if (info.description) {
                const help = document.createElement('div');
                help.className = 'form-help';
                help.textContent = info.description;
                group.appendChild(help);
            }
            
            // Add error container
            const error = document.createElement('div');
            error.className = 'form-error';
            group.appendChild(error);
            
            return group;
        }

        function validateForm() {
            const form = document.getElementById('tool_prompt_form');
            let isValid = true;
            
            // Clear previous errors
            document.querySelectorAll('.form-control').forEach(function(control) {
                control.classList.remove('error');
            });
            document.querySelectorAll('.form-error').forEach(function(error) {
                error.style.display = 'none';
            });
            
            if (!currentToolData) return false;
            
            const props = currentToolData.schema?.properties || {};
            const required = currentToolData.schema?.required || [];
            
            for (let param in props) {
                const info = props[param];
                const isRequired = required.includes(param);
                const control = form.querySelector('[name="' + param + '"]');
                
                if (!control) continue;
                
                let value = control.value;
                if (control.getArrayValue) {
                    value = control.getArrayValue();
                }
                
                // Check required fields
                if (isRequired && (!value || (Array.isArray(value) && value.length === 0))) {
                    control.classList.add('error');
                    const errorDiv = control.closest('.form-group').querySelector('.form-error');
                    errorDiv.textContent = 'This field is required';
                    errorDiv.style.display = 'block';
                    isValid = false;
                }
                
                // Type validation
                if (value && info.type === 'integer') {
                    const numValue = parseInt(value);
                    if (isNaN(numValue)) {
                        control.classList.add('error');
                        const errorDiv = control.closest('.form-group').querySelector('.form-error');
                        errorDiv.textContent = 'Must be a valid number';
                        errorDiv.style.display = 'block';
                        isValid = false;
                    }
                }
            }
            
            return isValid;
        }

        function showToolPrompt(data) {
            currentToolData = data;
            const modal = document.getElementById('tool_prompt_modal');
            const form = document.getElementById('tool_prompt_form');
            const title = document.getElementById('tool_name_text');
            const description = document.getElementById('tool_description');
            const preview = document.getElementById('parameter_preview');
            const previewContent = document.getElementById('parameter_preview_content');
            
            // Update header
            const toolIcon = document.querySelector('.tool-icon');
            toolIcon.textContent = getToolIcon(data.tool);
            title.textContent = `Confirm ${data.tool}`;
            description.textContent = getToolDescription(data.tool, data.schema);
            
            // Show parameter preview if there are existing values
            if (data.values && Object.keys(data.values).length > 0) {
                previewContent.textContent = JSON.stringify(data.values, null, 2);
                preview.style.display = 'block';
            } else {
                preview.style.display = 'none';
            }
            
            // Clear and rebuild form
            form.innerHTML = '';
            
            const props = data.schema?.properties || {};
            const required = data.schema?.required || [];
            
            for (let param in props) {
                const info = { ...props[param], required: required.includes(param) };
                const value = data.values?.[param];
                const field = createFormField(param, info, value);
                form.appendChild(field);
            }
            
            // Reset button state
            const executeBtn = document.getElementById('execute_btn');
            const executeText = document.getElementById('execute_text');
            const loadingSpinner = document.getElementById('loading_spinner');
            
            executeBtn.disabled = false;
            executeText.style.display = 'inline';
            loadingSpinner.style.display = 'none';
            
            modal.style.display = 'block';
            
            // Focus the first interactive input (text, checkbox, radio, select, textarea)
            setTimeout(() => {
                const focusable = form.querySelectorAll(
                    'input:not([type=hidden]):not([disabled]), select:not([disabled]), textarea:not([disabled])'
                );
                for (const el of focusable) {
                    // Skip elements that are not visible
                    if (el.offsetParent !== null) {
                        el.focus();
                        break;
                    }
                }
            }, 100);
        }

        function hideToolPrompt() {
            document.getElementById('tool_prompt_modal').style.display = 'none';
            currentToolData = null;
        }

        function cancelToolCall() {
            hideToolPrompt();
            // Optionally emit cancellation event
            socket.emit('tool_response', { cancelled: true });
        }

        function executeToolCall() {
            if (!validateForm()) {
                return;
            }
            
            const form = document.getElementById('tool_prompt_form');
            const executeBtn = document.getElementById('execute_btn');
            const executeText = document.getElementById('execute_text');
            const loadingSpinner = document.getElementById('loading_spinner');
            
            // Show loading state
            executeBtn.disabled = true;
            executeText.style.display = 'none';
            loadingSpinner.style.display = 'inline-block';
            
            const result = {};
            const props = currentToolData.schema?.properties || {};
            
            for (let param in props) {
                const info = props[param];
                const control = form.querySelector('[name="' + param + '"]');
                
                if (!control) continue;
                
                let value = control.value;
                
                // Handle different input types
                if (control.type === 'checkbox') {
                    value = control.checked;
                } else if (control.getArrayValue) {
                    value = control.getArrayValue();
                } else if (info.type === 'integer') {
                    const parsed = parseInt(value);
                    if (!isNaN(parsed)) value = parsed;
                } else if (info.type === 'array' && typeof value === 'string') {
                    try {
                        value = JSON.parse(value);
                    } catch (err) {
                        value = value.split(',').map(v => v.trim()).filter(v => v);
                    }
                }
                
                result[param] = value;
            }
            
            socket.emit('tool_response', { input: result });
            
            // Modal will be hidden by the tool_prompt_clear event
        }

        function sendMessage() {
            var input = document.getElementById('user_input');
            var message = input.value;
            if (message.trim() !== '') {
                socket.emit('user_input', { input: message });
                input.value = '';
            }
        }

        function interruptAgent() {
            socket.emit('interrupt');
            alert('Agent interruption requested.');
        }

        // Handle Enter key in textarea
        document.getElementById('user_input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Handle Escape key to close modal
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && document.getElementById('tool_prompt_modal').style.display === 'block') {
                cancelToolCall();
            }
        });

        // Handle click outside modal to close
        document.getElementById('tool_prompt_modal').addEventListener('click', function(e) {
            if (e.target === this) {
                cancelToolCall();
            }
        });

        // Fetch initial messages when page loads
        window.onload = function() {
            fetch('/messages')
                .then(response => response.json())
                .then(data => {
                    updateMessages(data);
                });
        };
    

});
