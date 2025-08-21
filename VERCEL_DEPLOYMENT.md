# Vercel Deployment Guide for Slazy

This guide explains how to deploy the Slazy application to Vercel, including troubleshooting common issues.

## Prerequisites

1. A Vercel account (sign up at https://vercel.com)
2. Vercel CLI installed: `npm i -g vercel`
3. Git repository with your code

## Deployment Files Created

The following files have been created/modified for Vercel deployment:

1. **`vercel.json`** - Vercel configuration file
2. **`api/index.py`** - Serverless function entry point
3. **`requirements-vercel.txt`** - Simplified dependencies for Vercel

## Deployment Steps

### Option 1: Deploy via Vercel CLI

1. Install Vercel CLI if not already installed:
   ```bash
   npm i -g vercel
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

3. Deploy the application:
   ```bash
   vercel --prod
   ```

### Option 2: Deploy via GitHub Integration

1. Push your code to GitHub
2. Go to https://vercel.com/new
3. Import your GitHub repository
4. Vercel will auto-detect the configuration and deploy

## Environment Variables

You need to set the following environment variables in Vercel:

1. Go to your project settings in Vercel Dashboard
2. Navigate to "Environment Variables"
3. Add the following:
   - `OPENROUTER_API_KEY` or `OPENAI_API_KEY`
   - `OPENAI_BASE_URL` (optional, for OpenRouter)

## Important Limitations and Considerations

### 1. WebSocket Support
**Issue**: Vercel serverless functions don't support WebSockets natively.
**Impact**: The Flask-SocketIO functionality won't work as expected.
**Solution**: 
- Consider using Server-Sent Events (SSE) instead
- Or deploy to a platform that supports WebSockets (Railway, Render, Heroku)

### 2. Serverless Function Timeout
**Issue**: Vercel has a 10-second timeout for hobby accounts, 60 seconds for Pro.
**Impact**: Long-running agent tasks will timeout.
**Solution**: 
- Break down tasks into smaller chunks
- Use background jobs with a queue system
- Consider upgrading to Vercel Pro for longer timeouts

### 3. File System Limitations
**Issue**: Vercel's serverless functions have read-only file systems (except `/tmp`).
**Impact**: Cannot write logs or create files persistently.
**Solution**: 
- Use external storage (S3, database) for persistent data
- Modify code to use `/tmp` for temporary files

### 4. Cold Starts
**Issue**: Serverless functions have cold start delays.
**Impact**: First request after inactivity will be slow.
**Solution**: 
- Keep functions warm with scheduled pings
- Optimize import statements and initialization

## Troubleshooting Common Issues

### Error: "Module not found"
**Cause**: Missing dependencies in requirements-vercel.txt
**Fix**: Add the missing module to requirements-vercel.txt

### Error: "Function timeout"
**Cause**: Agent task taking too long
**Fix**: 
- Increase timeout in vercel.json (max 60s for Pro)
- Optimize the task processing
- Consider alternative deployment platforms

### Error: "Cannot import name 'X' from 'Y'"
**Cause**: Import path issues due to different directory structure
**Fix**: Check the sys.path configuration in api/index.py

### Error: "Permission denied" when writing files
**Cause**: Trying to write to read-only file system
**Fix**: Modify code to use `/tmp` directory or external storage

## Alternative Deployment Platforms

If Vercel's limitations are too restrictive, consider these alternatives:

### For WebSocket Support:
- **Railway** (railway.app) - Full Python support with WebSockets
- **Render** (render.com) - Supports Flask with WebSockets
- **Fly.io** (fly.io) - Docker-based deployment with WebSocket support

### For Long-Running Tasks:
- **Heroku** - Up to 30-minute request timeout
- **Google Cloud Run** - Up to 60-minute timeout
- **AWS EC2/ECS** - No timeout limitations

## Modified Deployment for Better Vercel Compatibility

To make the app more Vercel-friendly, consider these modifications:

1. **Replace WebSockets with SSE**:
   - Server-Sent Events work with serverless
   - One-way communication from server to client

2. **Use External Queue Service**:
   - Redis + Celery for background tasks
   - AWS SQS for job queuing

3. **External Storage**:
   - Use S3 for file storage
   - PostgreSQL/MongoDB for data persistence

## Testing Locally

Before deploying, test the Vercel function locally:

```bash
# Install Vercel CLI
npm i -g vercel

# Run locally
vercel dev
```

This will simulate the Vercel environment locally.

## Monitoring and Logs

1. View function logs in Vercel Dashboard → Functions tab
2. Use Vercel Analytics for performance monitoring
3. Set up error tracking with Sentry integration

## Next Steps

1. Deploy using one of the methods above
2. Set environment variables in Vercel Dashboard
3. Test the deployment
4. Monitor logs for any issues

If you encounter issues not covered here, check:
- Vercel documentation: https://vercel.com/docs
- Vercel Python examples: https://github.com/vercel/examples/tree/main/python

## Summary of Changes Made

1. Created `vercel.json` with Python runtime configuration
2. Created `api/index.py` as serverless function wrapper
3. Created `requirements-vercel.txt` with minimal dependencies
4. Configured static file serving for `/public/static`

The main challenge is that this application uses WebSockets (Flask-SocketIO), which aren't supported by Vercel's serverless architecture. For full functionality, consider deploying to a platform that supports persistent connections.