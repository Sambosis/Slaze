# Console Markdown Block Rendering Fix

## Issue Description
The envsetup tool was passing console markdown blocks to the display assistant, but these blocks were not being rendered as code blocks on the webpage. Instead, they were being displayed as plain text due to HTML escaping.

## Root Cause
The issue was in `templates/index.html` where assistant messages were being processed with `escapeHtml(msg)` instead of parsing markdown:

```javascript
// Before (line 754):
assistantMessagesDiv.innerHTML += '<div class="message assistant-message">' + escapeHtml(msg) + '</div>';

// After:
assistantMessagesDiv.innerHTML += '<div class="message assistant-message">' + parseMarkdown(msg) + '</div>';
```

## Solution Implementation

### 1. Added Markdown and Syntax Highlighting Libraries
Added the following CDN links to the HTML head:
- `marked.js` - for markdown parsing
- `prism.js` - for syntax highlighting  
- `prism-bash.js` - for bash/shell highlighting
- `prism-shell-session.js` - for shell session highlighting
- `prism.css` - for syntax highlighting styles

### 2. Created Markdown Parser Function
Added `parseMarkdown()` function that:
- Handles type conversion (similar to `escapeHtml`)
- Configures marked.js to use Prism for syntax highlighting
- Maps `console` language to `shell-session` or `bash` highlighting
- Returns properly formatted HTML

### 3. Updated Assistant Message Handling
Changed assistant message processing from HTML escaping to markdown parsing.

### 4. Added CSS Styles
Added comprehensive CSS styles for:
- Code blocks (`<pre>` and `<code>`)
- Syntax highlighting themes
- Console/shell-specific styling with dark theme
- Proper typography and spacing

### 5. Added Dynamic Re-highlighting
Added `Prism.highlightAll()` call after message updates to ensure syntax highlighting is applied to dynamically inserted content.

## Files Modified
- `templates/index.html` - Updated JavaScript and CSS for markdown rendering

## Expected Behavior
Console markdown blocks from envsetup (like `\`\`\`console ... \`\`\``) should now render as properly formatted and syntax-highlighted code blocks on the webpage instead of plain text.

## Example
Console blocks like this from envsetup:
```
```console
$ cd /workspace
$ python -m venv venv
$ source venv/bin/activate
[Exit code: 0]
```
```

Should now render as dark-themed, syntax-highlighted code blocks with proper shell/console formatting.