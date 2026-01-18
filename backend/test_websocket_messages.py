#!/usr/bin/env python3
"""
Test WebSocket message handling
"""
import asyncio
import json
import websockets

async def test_websocket_messages():
    """Test sending different message types to the WebSocket"""
    uri = "ws://localhost:8000/ws/projects/demo-project"
    
    async with websockets.connect(uri) as websocket:
        print("✓ Connected to WebSocket")
        
        # Receive HELLO message
        hello = await websocket.recv()
        print(f"\n📨 Received HELLO: {json.loads(hello)['event']['name']}")
        
        # Receive initial STAGE_STATUS
        status = await websocket.recv()
        print(f"📨 Received STAGE_STATUS: {json.loads(status)['event']['name']}")
        
        # Test 1: Send a chat message
        print("\n🧪 Test 1: Sending chat message...")
        chat_msg = {
            "type": "chat",
            "text": "Hello from WebSocket client! Please optimize for accuracy."
        }
        await websocket.send(json.dumps(chat_msg))
        
        # Wait for response (LOG_LINE event)
        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        resp_data = json.loads(response)
        print(f"✓ Received response: {resp_data['event']['name']}")
        print(f"  Payload: {resp_data['event']['payload']}")
        
        # Test 2: Send a ping
        print("\n🧪 Test 2: Sending ping...")
        ping_msg = {
            "type": "ping",
            "ts": 123456789
        }
        await websocket.send(json.dumps(ping_msg))
        
        # Wait for pong
        pong = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        pong_data = json.loads(pong)
        print(f"✓ Received pong: {pong_data}")
        
        # Test 3: Send a confirm message
        print("\n🧪 Test 3: Sending confirm...")
        confirm_msg = {
            "type": "confirm"
        }
        await websocket.send(json.dumps(confirm_msg))
        
        # Wait for stage update
        stage_update = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        stage_data = json.loads(stage_update)
        print(f"✓ Received stage update: {stage_data['event']['name']}")
        print(f"  New stage: {stage_data['event']['payload']}")
        
        # Test 4: Send a command
        print("\n🧪 Test 4: Sending restart_stage command...")
        command_msg = {
            "type": "command",
            "command": "restart_stage",
            "args": {
                "stage_id": "PARSE_INTENT"
            }
        }
        await websocket.send(json.dumps(command_msg))
        
        # Wait for response
        cmd_response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
        cmd_data = json.loads(cmd_response)
        print(f"✓ Received command response: {cmd_data['event']['name']}")
        
        print("\n✅ All tests passed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_websocket_messages())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
