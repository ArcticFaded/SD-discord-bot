# Temporal Workflow Setup for SD Discord Bot

This bot now uses Temporal for robust queue management and workflow orchestration, replacing the previous custom queue system.

## Architecture Overview

- **Temporal Server**: Manages workflow state and task queues
- **Channel Workflows**: Each Discord channel gets its own workflow instance
- **Activities**: Individual tasks like image generation and Discord messaging
- **Worker**: Processes workflow tasks and activities

## Quick Start

### 1. Start Temporal Server

```bash
docker-compose up -d
```

This starts:
- Temporal server on port 7233
- Temporal Web UI on port 8080 (http://localhost:8080)
- PostgreSQL database for Temporal state

### 2. Start the Temporal Worker

In a separate terminal:

```bash
python temporal_worker.py
```

This worker:
- Connects to Temporal server
- Starts the main ChannelManagerWorkflow
- Processes image generation activities
- Handles Discord status updates

### 3. Start the Discord Bot

```bash
python run.py
```

## How It Works

### Request Flow

1. User submits `/generate` command in Discord
2. Bot defers response and gets followup webhook URL
3. Request is sent to Temporal's ChannelManagerWorkflow
4. Channel-specific workflow is created/reused
5. Multiple status updates are sent:
   - "✅ Request queued! Position: X"
   - "🎨 Processing your request..."
   - "⚡ Generating image..."
   - "✅ Image generated successfully!"
   - Final image with embed

### Features

- **Per-Channel Queues**: Each channel has independent queue
- **User Limits**: Max 2 concurrent requests per user
- **Progress Updates**: Multiple status messages during generation
- **Retry Logic**: Automatic retries on failures
- **15-minute Window**: Can send updates for 15 minutes after initial response

### Configuration

The system uses existing `config.json` for:
- Discord bot token
- Channel-to-API server mapping
- Image generation defaults

### Monitoring

Access Temporal Web UI at http://localhost:8080 to:
- View active workflows
- Monitor queue status
- Debug failed activities
- See workflow history

## Benefits Over Previous System

1. **Reliability**: Workflows persist across restarts
2. **Scalability**: Handles 5000+ requests/day easily
3. **Visibility**: Web UI for monitoring and debugging
4. **Multiple Updates**: Send progress updates to users
5. **Error Handling**: Built-in retry and error recovery
6. **Distributed**: Can scale workers horizontally

## Troubleshooting

### Worker Not Connecting
- Ensure Temporal server is running: `docker-compose ps`
- Check logs: `docker-compose logs temporal`

### Workflows Not Processing
- Verify worker is running: Check worker terminal output
- Check Temporal UI for workflow status

### Discord Not Receiving Messages
- Check interaction token is valid (15-minute window)
- Verify followup URL format in logs

## Advanced Configuration

### Scaling Workers

Run multiple workers for higher throughput:

```bash
# Terminal 1
python temporal_worker.py

# Terminal 2  
python temporal_worker.py
```

### Custom Timeouts

Edit `temporal_workflows.py`:
- Image generation: 180 seconds default
- Discord updates: 30 seconds default

### Queue Limits

Edit `ImageGenerationWorkflow` in `temporal_workflows.py`:
- `max_concurrent_per_user`: User request limit (default: 2)
- Workflow timeout: 24 hours of inactivity