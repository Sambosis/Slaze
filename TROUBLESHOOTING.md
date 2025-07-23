# Troubleshooting Guide

## Error: Custom Element 'mce-autosize-textarea' Already Defined

### Problem

Browser shows errors like:

```javascript
Uncaught Error: A custom element with name 'mce-autosize-textarea' has already been defined.
Failed to load resource: the server responded with a status of 500 (INTERNAL SERVER ERROR)
```

### Root Cause

This error typically occurs when:

1. Multiple instances of the same web component library are loaded
2. Browser extensions are interfering with custom elements
3. Previous page loads haven't properly cleaned up custom elements
4. Development tools or other applications are injecting scripts

### Solutions

#### Solution 1: Browser Cleanup

1. **Clear browser cache completely**:
   - Press `Ctrl+Shift+Delete`
   - Select "All time" and clear everything
   - Restart browser

2. **Disable browser extensions**:
   - Open browser in incognito/private mode
   - Or disable all extensions temporarily

3. **Try a different browser**:
   - Test with Chrome, Firefox, or Edge
   - Use a fresh browser profile

#### Solution 2: Check for Conflicting Applications

1. **Close other development servers**:
   - Stop any other Flask/web applications
   - Check for applications using port 5002
   - Use `netstat -ano | findstr :5002` to check

2. **Check for browser development tools**:
   - Close any open browser developer tools
   - Disable any live-reload extensions

#### Solution 3: Server-Side Fixes

1. **Restart the Slazy web server**:

   ```powershell
   # Stop current server (Ctrl+C)
   # Then restart:
   python restart_server.py  # Uses the new restart script
   ```

2. **Clear application cache**:
   - Delete `cache/` directory contents
   - Restart application

#### Solution 4: Code-Level Fixes

If the issue persists, it might be related to how custom elements are being loaded or defined in the templates.

### Prevention

1. Always use incognito mode for testing
2. Keep browser extensions minimal during development
3. Use different ports for different projects
4. Clear cache regularly during development

### Quick Fix Script

Use the provided `restart_server.py` script:

```powershell
python restart_server.py
```

This script will:
- Clear application cache
- Find an available port
- Start the server with helpful tips
- Provide troubleshooting guidance

### Applied Fixes

The following improvements have been made to prevent these errors:

1. **Enhanced error handling** in `utils/web_ui.py`
2. **Custom element conflict prevention** in templates
3. **Server restart script** (`restart_server.py`)
4. **Better logging** for debugging server issues
