import asyncio
from typing import Optional, Dict, Any
from temporalio.client import Client, WorkflowHandle
from temporal_activities import ImageGenerationRequest
from temporal_workflows import ChannelManagerWorkflow
import logging

logger = logging.getLogger(__name__)

class TemporalClient:
    """Client for interacting with Temporal workflows"""
    
    def __init__(self, host: str = "localhost:7233", namespace: str = "default"):
        self.host = host
        self.namespace = namespace
        self._client: Optional[Client] = None
        self._manager_handle: Optional[WorkflowHandle] = None
        self._lock = asyncio.Lock()
    
    async def connect(self):
        """Connect to Temporal and get manager workflow handle"""
        async with self._lock:
            if self._client is None:
                self._client = await Client.connect(self.host, namespace=self.namespace)
                
                # Get handle to the manager workflow
                self._manager_handle = self._client.get_workflow_handle(
                    workflow_id="channel-manager-main"
                )
                logger.info("Connected to Temporal")
    
    async def ensure_connected(self):
        """Ensure we're connected to Temporal"""
        if self._client is None:
            await self.connect()
    
    async def submit_image_request(
        self,
        channel_id: str,
        followup_url: str,
        author_name: str,
        author_id: str,
        options: Dict[str, Any],
        embed_data: Dict[str, Any]
    ) -> bool:
        """Submit an image generation request to Temporal"""
        try:
            await self.ensure_connected()
            
            # Create the request
            request = ImageGenerationRequest(
                channel_id=channel_id,
                inter_followup_url=followup_url,
                author_name=f"{author_name}#{author_id}",  # Include ID for uniqueness
                options=options,
                embed_data=embed_data
            )
            
            # Send to manager workflow
            await self._manager_handle.signal(
                ChannelManagerWorkflow.route_request,
                request
            )
            
            logger.info(f"Submitted request for channel {channel_id}, user {author_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting request to Temporal: {e}")
            return False
    
    async def get_queue_status(self, channel_id: str) -> Optional[Dict]:
        """Get queue status for a specific channel"""
        try:
            await self.ensure_connected()
            
            # Get handle to channel workflow
            workflow_id = f"image-gen-channel-{channel_id}"
            handle = self._client.get_workflow_handle(workflow_id=workflow_id)
            
            # Query the workflow
            result = await handle.query("get_queue_status")
            return result
            
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return None
    
    async def get_queue_position(self, channel_id: str, user_id: str) -> int:
        """Get queue position for a specific user in a channel"""
        try:
            await self.ensure_connected()
            
            workflow_id = f"image-gen-channel-{channel_id}"
            handle = self._client.get_workflow_handle(workflow_id=workflow_id)
            
            result = await handle.query("get_queue_position", user_id)
            return result
            
        except Exception as e:
            logger.error(f"Error getting queue position: {e}")
            return -1
    
    async def get_active_channels(self) -> Optional[Dict[str, str]]:
        """Get all active channel workflows"""
        try:
            await self.ensure_connected()
            
            result = await self._manager_handle.query("get_active_channels")
            return result
            
        except Exception as e:
            logger.error(f"Error getting active channels: {e}")
            return None

# Global client instance
temporal_client = TemporalClient()