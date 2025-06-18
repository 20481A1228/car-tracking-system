# test_websocket.py - Run this script to test WebSocket functionality

import asyncio
import websockets
import json
import sys

async def test_websocket_connection():
    """Test WebSocket connection to the car tracking system."""
    
    print("🔍 Testing WebSocket connections...")
    print("=" * 50)
    
    # Test URLs
    test_urls = [
        "ws://localhost:8000/ws/stream/1/",
        "ws://localhost:8000/ws/tracking/1/", 
        "ws://localhost:8000/ws/live-updates/"
    ]
    
    for url in test_urls:
        try:
            print(f"\n🔗 Testing: {url}")
            
            async with websockets.connect(url) as websocket:
                print(f"✅ Connected successfully!")
                
                # Send a test message
                test_message = {
                    "type": "ping",
                    "timestamp": "test"
                }
                
                await websocket.send(json.dumps(test_message))
                print(f"📤 Sent test message")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📥 Received: {response}")
                except asyncio.TimeoutError:
                    print(f"⏰ No response received (timeout)")
                
                print(f"✅ Connection test passed!")
                
        except ConnectionRefusedError:
            print(f"❌ Connection refused - server not running?")
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 WebSocket testing complete!")

def check_dependencies():
    """Check if all required packages are installed."""
    
    print("📦 Checking dependencies...")
    print("-" * 30)
    
    required_packages = [
        'django',
        'channels', 
        'channels_redis',
        'redis',
        'websockets',
        'daphne'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print(f"💡 Install with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"\n✅ All dependencies installed!")
        return True

def check_redis_connection():
    """Check if Redis is running and accessible."""
    
    print("\n🔴 Checking Redis connection...")
    print("-" * 30)
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis is running and accessible!")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {str(e)}")
        print("💡 Make sure Redis server is running: redis-server")
        return False

def main():
    """Main test function."""
    
    print("🚗 Car Tracking WebSocket Test Suite")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check Redis
    redis_ok = check_redis_connection()
    
    if not deps_ok or not redis_ok:
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Test WebSocket connections
    print("\n🌐 Testing WebSocket connections...")
    print("Make sure your Django server is running: python manage.py runserver")
    input("Press Enter when server is ready...")
    
    try:
        asyncio.run(test_websocket_connection())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main()

# Alternative: Simple synchronous test
def simple_test():
    """Simple synchronous test without asyncio."""
    
    print("🔧 Simple WebSocket Test")
    print("-" * 30)
    
    try:
        import websocket
        
        def on_message(ws, message):
            print(f"📥 Received: {message}")
        
        def on_error(ws, error):
            print(f"❌ Error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"🔌 Connection closed")
        
        def on_open(ws):
            print(f"✅ WebSocket connection opened!")
            ws.send('{"type":"ping","timestamp":"test"}')
        
        # Test streaming WebSocket
        ws = websocket.WebSocketApp("ws://localhost:8000/ws/stream/1/",
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
        
        print("🔗 Connecting to WebSocket...")
        ws.run_forever()
        
    except ImportError:
        print("❌ websocket-client not installed")
        print("💡 Install with: pip install websocket-client")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

# Uncomment this line to run simple test instead:
simple_test()