from datetime import timedelta
from typing import Dict, List, Optional
from temporalio import workflow
from temporalio.common import RetryPolicy
import asyncio

with workflow.unsafe.imports_passed_through():
    from temporal_activities import (
        ImageGenerationRequest,
        ImageGenerationResult,
        generate_image,
        send_discord_response,
        send_discord_status,
        check_user_request_limit
    )

@workflow.defn
class ImageGenerationWorkflow:
    """Workflow for handling image generation requests per channel"""
    
    def __init__(self):
        self.pending_requests: List[ImageGenerationRequest] = []
        self.active_users: Dict[str, int] = {}  # Track active requests per user
        self.processing = False
        self.max_concurrent_per_user = 2
    
    @workflow.run
    async def run(self, channel_id: str) -> None:
        """Main workflow run - keeps running and processing requests for a channel"""
        workflow.logger.info(f"Starting ImageGenerationWorkflow for channel {channel_id}")
        
        # Keep the workflow running
        while True:
            # Wait for signal or timeout
            await workflow.wait_condition(
                lambda: len(self.pending_requests) > 0,
                timeout=timedelta(hours=24)  # Workflow stays alive for 24 hours of inactivity
            )
            
            if len(self.pending_requests) > 0:
                await self.process_next_request()
    
    async def process_next_request(self):
        """Process the next request in queue"""
        if not self.pending_requests or self.processing:
            return
        
        request = self.pending_requests.pop(0)
        self.processing = True
        
        try:
            # Send initial status update - Processing started
            await workflow.execute_activity(
                send_discord_status,
                args=[
                    request.inter_followup_url,
                    f"🎨 **Processing your request**\nPrompt: `{request.options.get('prompt', '')[:100]}{'...' if len(request.options.get('prompt', '')) > 100 else ''}`",
                    None,
                    True
                ],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2)
            )
            
            # Send status update - Generating
            await workflow.execute_activity(
                send_discord_status,
                args=[
                    request.inter_followup_url,
                    f"⚡ **Generating image...**\n• Size: {request.options.get('width', 512)}x{request.options.get('height', 512)}\n• Steps: {request.options.get('steps', 20)}\n• Sampler: {request.options.get('sampler_index', 'Euler')}",
                    None,
                    True
                ],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2)
            )
            
            # Generate the image
            result = await workflow.execute_activity(
                generate_image,
                request,
                start_to_close_timeout=timedelta(seconds=180),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=10),
                    backoff_coefficient=2
                )
            )
            
            # Send final status before image
            if result.success:
                await workflow.execute_activity(
                    send_discord_status,
                    args=[
                        request.inter_followup_url,
                        f"✅ **Image generated successfully!**\nSeed: {result.parameters.get('seed', 'N/A') if result.parameters else 'N/A'}",
                        None,
                        True
                    ],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=2)
                )
            
            # Send response back to Discord
            await workflow.execute_activity(
                send_discord_response,
                args=[request.inter_followup_url, result, True],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3)
            )
            
            # Update user count
            user_id = request.author_name
            if user_id in self.active_users:
                self.active_users[user_id] = max(0, self.active_users[user_id] - 1)
                if self.active_users[user_id] == 0:
                    del self.active_users[user_id]
            
        except Exception as e:
            workflow.logger.error(f"Error processing request: {e}")
            # Try to send error message
            error_result = ImageGenerationResult(
                success=False,
                error=str(e)
            )
            try:
                await workflow.execute_activity(
                    send_discord_response,
                    args=[request.inter_followup_url, error_result, True],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=1)
                )
            except:
                pass  # Best effort error reporting
        
        finally:
            self.processing = False
    
    @workflow.signal
    async def submit_request(self, request: ImageGenerationRequest) -> None:
        """Signal to submit a new image generation request"""
        user_id = request.author_name
        
        # Check user limit
        user_count = self.active_users.get(user_id, 0)
        if user_count >= self.max_concurrent_per_user:
            # Send rejection message
            workflow.create_task(
                workflow.execute_activity(
                    send_discord_status,
                    args=[
                        request.inter_followup_url,
                        "❌ **Request limit reached**\nYou already have 2 requests in the queue. Please wait for them to complete.",
                        None,
                        True
                    ],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=1)
                )
            )
            return
        
        # Add to queue
        self.pending_requests.append(request)
        self.active_users[user_id] = user_count + 1
        
        # Send queued message with position
        queue_position = len(self.pending_requests)
        workflow.create_task(
            workflow.execute_activity(
                send_discord_status,
                args=[
                    request.inter_followup_url,
                    f"✅ **Request queued!**\n📍 Position in queue: **{queue_position}**\n⏱️ Estimated wait: ~{queue_position * 30} seconds",
                    None,
                    True
                ],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2)
            )
        )
        
        workflow.logger.info(f"Request queued at position {queue_position} for user {user_id}")
    
    @workflow.query
    def get_queue_status(self) -> Dict:
        """Query to get current queue status"""
        return {
            "queue_length": len(self.pending_requests),
            "processing": self.processing,
            "active_users": dict(self.active_users)
        }
    
    @workflow.query  
    def get_queue_position(self, user_id: str) -> int:
        """Get queue position for a specific user"""
        for i, req in enumerate(self.pending_requests):
            if req.author_name == user_id:
                return i + 1
        return -1

@workflow.defn
class ChannelManagerWorkflow:
    """Manager workflow that routes requests to channel-specific workflows"""
    
    def __init__(self):
        self.active_channels: Dict[str, str] = {}  # channel_id -> workflow_id mapping
    
    @workflow.run
    async def run(self) -> None:
        """Keep the manager running"""
        workflow.logger.info("ChannelManagerWorkflow started")
        # This workflow runs indefinitely
        await workflow.wait_condition(lambda: False)
    
    @workflow.signal
    async def route_request(self, request: ImageGenerationRequest) -> None:
        """Route request to appropriate channel workflow"""
        channel_id = request.channel_id
        
        # Check if we have a workflow for this channel
        if channel_id not in self.active_channels:
            # Start a new workflow for this channel
            workflow_id = f"image-gen-channel-{channel_id}"
            self.active_channels[channel_id] = workflow_id
            
            # Start child workflow
            await workflow.start_child_workflow(
                ImageGenerationWorkflow.run,
                args=[channel_id],
                id=workflow_id,
                parent_close_policy=workflow.ParentClosePolicy.ABANDON
            )
        
        # Send signal to channel workflow
        workflow_handle = workflow.get_external_workflow_handle_for(
            ImageGenerationWorkflow.run,
            workflow_id=self.active_channels[channel_id]
        )
        await workflow_handle.signal(ImageGenerationWorkflow.submit_request, request)
    
    @workflow.query
    def get_active_channels(self) -> Dict[str, str]:
        """Get all active channel workflows"""
        return dict(self.active_channels)