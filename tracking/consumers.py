# tracking/consumers.py - COMPLETE WITH STABILITY FEATURES

import json
import asyncio
from datetime import datetime
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

class TrackingConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for general tracking updates with enhanced stability."""
    
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.session_group_name = f'tracking_{self.session_id}'
        self.last_heartbeat = datetime.now()

        await self.channel_layer.group_add(
            self.session_group_name,
            self.channel_name
        )

        await self.accept()
        print(f"🔗 Tracking WebSocket connected for session {self.session_id}")
        
        # Send welcome message
        await self.send(text_data=json.dumps({
            'type': 'connected',
            'message': f'Tracking connected for session {self.session_id}',
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.session_group_name,
            self.channel_name
        )
        print(f"🔌 Tracking WebSocket disconnected for session {self.session_id} (code: {close_code})")

    async def receive(self, text_data):
        """Handle incoming WebSocket messages with improved stability."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            # Update heartbeat timestamp
            self.last_heartbeat = datetime.now()

            if message_type == 'ping':
                # CRITICAL: Respond to heartbeat immediately
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': data.get('timestamp'),
                    'session_id': self.session_id,
                    'server_time': datetime.now().isoformat(),
                    'status': 'alive'
                }))
                
            elif message_type == 'get_live_data':
                live_data = await self.get_live_data()
                await self.send(text_data=json.dumps({
                    'type': 'live_data',
                    'data': live_data,
                    'timestamp': datetime.now().isoformat()
                }))
                
            elif message_type == 'keep_alive':
                # Additional keepalive handler
                await self.send(text_data=json.dumps({
                    'type': 'alive',
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'last_heartbeat': self.last_heartbeat.isoformat()
                }))
                
            elif message_type == 'subscribe':
                # Subscribe to specific updates
                await self.send(text_data=json.dumps({
                    'type': 'subscribed',
                    'message': 'Subscribed to tracking updates',
                    'session_id': self.session_id
                }))
                
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON data',
                'session_id': self.session_id
            }))
        except Exception as e:
            print(f"Tracking WebSocket receive error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing message: {str(e)}',
                'session_id': self.session_id
            }))

    async def tracking_update(self, event):
        """Handle tracking updates from background processing."""
        await self.send(text_data=json.dumps({
            'type': 'tracking_update',
            'data': event['data'],
            'timestamp': datetime.now().isoformat()
        }))

    @database_sync_to_async
    def get_live_data(self):
        """Get live tracking data from database."""
        # Import models inside the function to avoid Django setup issues
        from .models import CarTrackingSession, Car
        
        try:
            session = CarTrackingSession.objects.get(id=self.session_id)
            cars = session.cars.filter(is_active=True)
            
            live_data = []
            for car in cars:
                latest_point = car.tracking_points.last()
                if latest_point:
                    live_data.append({
                        'car_id': car.car_id,
                        'latitude': float(latest_point.latitude),
                        'longitude': float(latest_point.longitude),
                        'speed': float(latest_point.speed),
                        'timestamp': latest_point.timestamp.isoformat(),
                        'motion_confidence': float(latest_point.motion_confidence),
                        'is_moving': latest_point.is_moving
                    })
            
            return {
                'cars': live_data,
                'session_status': session.status,
                'processed_frames': session.processed_frames,
                'total_frames': session.total_frames,
                'cars_detected': session.cars_detected,
                'active_cars': len(live_data)
            }
        except Exception as e:
            return {'error': str(e)}

class VideoStreamConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for live video streaming with enhanced stability."""
    
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.stream_group_name = f'streaming_{self.session_id}'
        self.last_heartbeat = datetime.now()
        self.heartbeat_task = None

        # Join streaming group
        await self.channel_layer.group_add(
            self.stream_group_name,
            self.channel_name
        )

        await self.accept()
        
        print(f"🎥 Video streaming WebSocket connected for session {self.session_id}")
        
        # Send initial connection message
        await self.send(text_data=json.dumps({
            'type': 'connected',
            'message': f'Video stream connected for session {self.session_id}',
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'heartbeat_interval': 30  # Tell client to send ping every 30s
        }))
        
        # Start server-side heartbeat monitoring
        self.heartbeat_task = asyncio.create_task(self.heartbeat_monitor())

    async def disconnect(self, close_code):
        # Cancel heartbeat monitoring
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            
        # Leave streaming group
        await self.channel_layer.group_discard(
            self.stream_group_name,
            self.channel_name
        )
        print(f"🔌 Video streaming WebSocket disconnected for session {self.session_id} (code: {close_code})")

    async def heartbeat_monitor(self):
        """Monitor client heartbeat and disconnect if no activity."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                # Check if last heartbeat was more than 2 minutes ago
                time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
                if time_since_heartbeat > 120:  # 2 minutes
                    print(f"⚠️ No heartbeat for {time_since_heartbeat}s, closing connection")
                    await self.close()
                    break
                    
        except asyncio.CancelledError:
            pass  # Task was cancelled, normal shutdown

    async def receive(self, text_data):
        """Handle incoming WebSocket messages with enhanced stability."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            # Update heartbeat timestamp for any received message
            self.last_heartbeat = datetime.now()

            if message_type == 'ping':
                # CRITICAL: Respond to heartbeat immediately
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': data.get('timestamp'),
                    'session_id': self.session_id,
                    'server_time': datetime.now().isoformat(),
                    'status': 'alive',
                    'connection_duration': (datetime.now() - self.last_heartbeat).total_seconds()
                }))
                
            elif message_type == 'start_stream':
                # Client is ready to receive stream
                await self.send(text_data=json.dumps({
                    'type': 'stream_ready',
                    'message': 'Ready to receive video stream',
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat()
                }))
                
            elif message_type == 'stop_stream':
                # Client wants to stop stream
                await self.send(text_data=json.dumps({
                    'type': 'stream_stop_request',
                    'message': 'Stream stop requested',
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat()
                }))
                
            elif message_type == 'keep_alive':
                # Additional keepalive handler
                await self.send(text_data=json.dumps({
                    'type': 'alive',
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'last_heartbeat': self.last_heartbeat.isoformat(),
                    'uptime': (datetime.now() - self.last_heartbeat).total_seconds()
                }))
                
            elif message_type == 'get_status':
                # Get current streaming status
                status = await self.get_streaming_status()
                await self.send(text_data=json.dumps({
                    'type': 'status_update',
                    'data': status,
                    'timestamp': datetime.now().isoformat()
                }))
                
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON data',
                'session_id': self.session_id
            }))
        except Exception as e:
            print(f"Video WebSocket receive error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing message: {str(e)}',
                'session_id': self.session_id
            }))

    async def stream_frame(self, event):
        """Send video frame with detection data to WebSocket client."""
        try:
            # Send the frame data and detection info
            await self.send(text_data=json.dumps({
                'type': 'video_frame',
                'data': {
                    'frame': event['frame'],
                    'moving_cars': event.get('moving_cars', 0),
                    'total_cars': event.get('total_cars', 0),
                    'fps': event.get('fps', 0),
                    'progress': event.get('progress', 0),
                    'timestamp': event.get('timestamp', ''),
                    'detections': event.get('detections', []),
                    'session_id': self.session_id,
                    'server_time': datetime.now().isoformat()
                }
            }))
            
        except Exception as e:
            print(f"Error sending stream frame: {e}")
            await self.send(text_data=json.dumps({
                'type': 'stream_error',
                'message': f'Error sending frame: {str(e)}',
                'session_id': self.session_id
            }))

    async def stream_started(self, event):
        """Notify client that streaming has started."""
        await self.send(text_data=json.dumps({
            'type': 'stream_started',
            'message': 'Video streaming started',
            'session_id': self.session_id,
            'data': event.get('data', {}),
            'timestamp': datetime.now().isoformat()
        }))

    async def stream_stopped(self, event):
        """Notify client that streaming has stopped."""
        await self.send(text_data=json.dumps({
            'type': 'stream_stopped',
            'message': 'Video streaming stopped',
            'session_id': self.session_id,
            'data': event.get('data', {}),
            'timestamp': datetime.now().isoformat()
        }))

    async def stream_error(self, event):
        """Send streaming error to client."""
        await self.send(text_data=json.dumps({
            'type': 'stream_error',
            'message': event.get('message', 'Unknown streaming error'),
            'error': event.get('error', 'Unknown error'),
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }))

    async def stream_status(self, event):
        """Send streaming status update to client."""
        await self.send(text_data=json.dumps({
            'type': 'stream_status',
            'data': event.get('data', {}),
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }))

    async def detection_update(self, event):
        """Send detection update to client."""
        await self.send(text_data=json.dumps({
            'type': 'detection_update',
            'data': event.get('data', {}),
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }))

    @database_sync_to_async
    def get_streaming_status(self):
        """Get current streaming status from database."""
        from .models import CarTrackingSession
        
        try:
            session = CarTrackingSession.objects.get(id=self.session_id)
            return {
                'session_status': session.status,
                'session_name': session.name,
                'cars_detected': session.cars_detected,
                'processed_frames': session.processed_frames,
                'total_frames': session.total_frames,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

class LiveUpdatesConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for general live updates with enhanced stability."""
    
    async def connect(self):
        self.last_heartbeat = datetime.now()
        
        await self.channel_layer.group_add('live_updates', self.channel_name)
        await self.accept()
        print("🔗 Live updates WebSocket connected")
        
        # Send welcome message
        await self.send(text_data=json.dumps({
            'type': 'connected',
            'message': 'Live updates connected',
            'timestamp': datetime.now().isoformat(),
            'heartbeat_interval': 30
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard('live_updates', self.channel_name)
        print(f"🔌 Live updates WebSocket disconnected (code: {close_code})")

    async def receive(self, text_data):
        """Handle incoming WebSocket messages with enhanced stability."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            # Update heartbeat timestamp
            self.last_heartbeat = datetime.now()

            if message_type == 'ping':
                # Respond to heartbeat
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': data.get('timestamp'),
                    'server_time': datetime.now().isoformat(),
                    'status': 'alive'
                }))
                
            elif message_type == 'subscribe':
                # Subscribe to specific updates
                await self.send(text_data=json.dumps({
                    'type': 'subscribed',
                    'message': 'Subscribed to live updates',
                    'timestamp': datetime.now().isoformat()
                }))
                
            elif message_type == 'keep_alive':
                await self.send(text_data=json.dumps({
                    'type': 'alive',
                    'timestamp': datetime.now().isoformat()
                }))
                
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON data'
            }))
        except Exception as e:
            print(f"Live updates WebSocket error: {e}")

    async def live_update(self, event):
        """Send live updates to client."""
        await self.send(text_data=json.dumps({
            'type': 'live_update',
            'data': event.get('data', {}),
            'timestamp': event.get('timestamp', datetime.now().isoformat())
        }))

    async def system_notification(self, event):
        """Send system notifications to client."""
        await self.send(text_data=json.dumps({
            'type': 'notification',
            'message': event.get('message', ''),
            'level': event.get('level', 'info'),
            'timestamp': event.get('timestamp', datetime.now().isoformat())
        }))

    async def session_update(self, event):
        """Send session status updates to client."""
        await self.send(text_data=json.dumps({
            'type': 'session_update',
            'data': event.get('data', {}),
            'session_id': event.get('session_id', ''),
            'timestamp': event.get('timestamp', datetime.now().isoformat())
        }))

    async def dashboard_update(self, event):
        """Send dashboard updates to client."""
        await self.send(text_data=json.dumps({
            'type': 'dashboard_update',
            'data': event.get('data', {}),
            'timestamp': event.get('timestamp', datetime.now().isoformat())
        }))

print("✅ ENHANCED CONSUMERS - With heartbeat monitoring and stability features")