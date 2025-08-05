import asyncio
import logging
from temporalio.client import Client
from temporalio.worker import Worker
from temporal_workflows import ImageGenerationWorkflow, ChannelManagerWorkflow
from temporal_activities import generate_image, send_discord_response, send_discord_status, check_user_request_limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_worker(temporal_host: str = "localhost:7233", namespace: str = "default"):
    """Run the Temporal worker"""
    # Create client
    client = await Client.connect(temporal_host, namespace=namespace)
    
    # Create worker with our workflows and activities
    worker = Worker(
        client,
        task_queue="sd-discord-bot",
        workflows=[ImageGenerationWorkflow, ChannelManagerWorkflow],
        activities=[generate_image, send_discord_response, send_discord_status, check_user_request_limit],
        max_concurrent_activity_task_executions=10,  # Limit concurrent image generations
        max_concurrent_workflow_task_executions=100,  # Handle many channels
    )
    
    logger.info("Starting Temporal worker...")
    await worker.run()

async def start_manager_workflow(temporal_host: str = "localhost:7233", namespace: str = "default"):
    """Start the main channel manager workflow if not already running"""
    client = await Client.connect(temporal_host, namespace=namespace)
    
    try:
        # Try to start the manager workflow
        handle = await client.start_workflow(
            ChannelManagerWorkflow.run,
            id="channel-manager-main",
            task_queue="sd-discord-bot",
        )
        logger.info(f"Started ChannelManagerWorkflow with ID: channel-manager-main")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("ChannelManagerWorkflow already running")
        else:
            logger.error(f"Error starting ChannelManagerWorkflow: {e}")
            raise

if __name__ == "__main__":
    async def main():
        # Start the manager workflow
        await start_manager_workflow()
        
        # Run the worker
        await run_worker()
    
    asyncio.run(main())