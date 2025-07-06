document.addEventListener('DOMContentLoaded', () => {
    const toolButtonsContainer = document.getElementById('tool-buttons-container');
    const commandFormContainer = document.getElementById('command-form-container');
    const resultsOutput = document.getElementById('results-output');

    // Define tools and their parameters
    const tools = {
        'ls': {
            args: [{ name: 'directory_path', type: 'text', default: '' }],
            description: 'Lists all files and directories under the given directory.'
        },
        'read_files': {
            args: [{ name: 'filepaths', type: 'textarea', placeholder: 'Enter filepaths, one per line' }],
            description: 'Returns the content of the specified files.'
        },
        'view_text_website': {
            args: [{ name: 'url', type: 'text' }],
            description: 'Fetches the content of a website as plain text.'
        },
        'set_plan': {
            args: [{ name: 'plan', type: 'textarea', placeholder: 'Enter plan details here...' }],
            description: 'Sets or updates the plan.'
        },
        'plan_step_complete': {
            args: [{ name: 'message', type: 'text' }],
            description: 'Marks the current plan step as complete.'
        },
        'message_user': {
            args: [
                { name: 'message', type: 'textarea' },
                { name: 'continue_working', type: 'checkbox', checked: true }
            ],
            description: 'Messages the user.'
        },
        'request_user_input': {
            args: [{ name: 'message', type: 'textarea' }],
            description: 'Asks the user a question and waits for a response.'
        },
        'submit': {
            args: [
                { name: 'branch_name', type: 'text' },
                { name: 'commit_message', type: 'textarea' }
            ],
            description: 'Commits the current code and requests user approval to push.'
        },
        'delete_file': {
            args: [{ name: 'filepath', type: 'text' }],
            description: 'Deletes a file.'
        },
        'rename_file': {
            args: [
                { name: 'filepath', type: 'text' },
                { name: 'new_filepath', type: 'text' }
            ],
            description: 'Renames and/or moves files and directories.'
        },
        'grep': {
            args: [{ name: 'pattern', type: 'text' }],
            description: 'Runs grep for the given pattern.'
        },
        'reset_all': {
            args: [],
            description: 'Resets the entire codebase to its original state.'
        },
        'restore_file': {
            args: [{ name: 'filepath', type: 'text' }],
            description: 'Restores the given file to its original state.'
        },
        'view_image': {
            args: [{ name: 'url', type: 'text' }],
            description: 'Downloads the image at the URL.'
        },
        'run_in_bash_session': {
            args: [{ name: 'command', type: 'textarea', placeholder: 'Enter bash command(s)'}],
            description: 'Runs the given bash command in the sandbox.',
            isDsl: true
        },
        'create_file_with_block': {
            args: [
                { name: 'filepath', type: 'text' },
                { name: 'content', type: 'textarea', placeholder: 'Enter file content' }
            ],
            description: 'Creates a new file with the given content.',
            isDsl: true
        },
        'overwrite_file_with_block': {
            args: [
                { name: 'filepath', type: 'text' },
                { name: 'content', type: 'textarea', placeholder: 'Enter new file content' }
            ],
            description: 'Overwrites an existing file with the given content.',
            isDsl: true
        },
        'replace_with_git_merge_diff': {
            args: [
                { name: 'filepath', type: 'text' },
                { name: 'search_block', type: 'textarea', placeholder: '<<<<<<< SEARCH\n...\n=======' },
                { name: 'replace_block', type: 'textarea', placeholder: '=======\n...\n>>>>>>> REPLACE' }
            ],
            description: 'Performs a targeted search-and-replace using git merge diff format.',
            isDsl: true
        }
    };

    function generateToolButtons() {
        toolButtonsContainer.innerHTML = '';
        for (const toolName in tools) {
            const button = document.createElement('button');
            button.textContent = toolName;
            button.addEventListener('click', () => displayCommandForm(toolName));
            toolButtonsContainer.appendChild(button);
        }
    }

    function displayCommandForm(toolName) {
        commandFormContainer.innerHTML = '';
        resultsOutput.textContent = ''; // Clear previous results

        const tool = tools[toolName];
        if (!tool) return;

        const form = document.createElement('form');
        form.id = 'tool-form';

        const title = document.createElement('h3');
        title.textContent = `Tool: ${toolName}`;
        form.appendChild(title);

        if (tool.description) {
            const description = document.createElement('p');
            description.textContent = tool.description;
            description.style.fontSize = '0.9em';
            description.style.color = '#555';
            form.appendChild(description);
        }

        tool.args.forEach(arg => {
            const argName = typeof arg === 'string' ? arg : arg.name;
            const argType = typeof arg === 'object' ? arg.type : 'text';
            const argPlaceholder = typeof arg === 'object' ? arg.placeholder : '';
            const argDefault = typeof arg === 'object' ? arg.default : undefined;
            const argChecked = typeof arg === 'object' ? arg.checked : undefined;

            const label = document.createElement('label');
            label.textContent = `${argName}:`;
            label.htmlFor = `param-${argName}`;
            form.appendChild(label);

            let input;
            if (argType === 'textarea') {
                input = document.createElement('textarea');
                if (argPlaceholder) input.placeholder = argPlaceholder;
            } else if (argType === 'checkbox') {
                input = document.createElement('input');
                input.type = 'checkbox';
                if (argChecked !== undefined) input.checked = argChecked;
            } else {
                input = document.createElement('input');
                input.type = argType;
                if (argPlaceholder) input.placeholder = argPlaceholder;
                if (argDefault !== undefined) input.value = argDefault;
            }
            input.id = `param-${argName}`;
            input.name = argName;
            form.appendChild(input);
            form.appendChild(document.createElement('br'));
        });

        const submitButton = document.createElement('button');
        submitButton.type = 'submit';
        submitButton.textContent = 'Run Tool';
        form.appendChild(submitButton);

        form.addEventListener('submit', (event) => {
            event.preventDefault();
            handleFormSubmit(toolName, form);
        });

        commandFormContainer.appendChild(form);
    }

    function handleFormSubmit(toolName, form) {
        const formData = new FormData(form);
        const toolDefinition = tools[toolName];
        let outputString = `Simulating execution of '${toolName}':\n\n`;

        outputString += '
