# Vercel Deployment Guide for Slazy Agent

## Overview

This repository has been configured for deployment on Vercel with the following modifications:

### Files Added/Modified:
- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless entry point
- `.vercelignore` - Files to exclude from deployment
- `requirements-vercel.txt` - Simplified dependencies for Vercel

## Important Limitations

**⚠️ WebSocket/SocketIO Limitations:**
The original application uses Flask-SocketIO for real-time communication, but Vercel's serverless platform has limitations with persistent WebSocket connections. The deployed version provides:

- Static file serving ✅
- Basic web interface ✅  
- API endpoints ✅
- Real-time agent execution ❌ (limited)

## Deployment Steps

1. **Connect to Vercel:**
   ```bash
   npm install -g vercel
   vercel login
   ```

2. **Deploy:**
   ```bash
   vercel --prod
   ```

3. **Environment Variables:**
   Set these in your Vercel dashboard:
   - `OPENAI_API_KEY`
   - Any other API keys your agents need

## Alternative Deployment Options

For full functionality including real-time WebSocket communication, consider:

1. **Railway** - Supports long-running processes and WebSockets
2. **Render** - Good for Flask-SocketIO applications  
3. **Heroku** - Traditional hosting with WebSocket support
4. **DigitalOcean App Platform** - Container-based deployment
5. **Google Cloud Run** - Containerized serverless with WebSocket support

## Local Development

The original functionality works perfectly locally:
```bash
python run.py web --port 5002
```

## Vercel-Specific Configuration

The `api/index.py` file creates a simplified Flask app that:
- Serves the web interface
- Provides basic API endpoints
- Handles static files
- Shows appropriate messaging about limitations

For production use with full agent capabilities, we recommend deploying to a platform that supports persistent connections.