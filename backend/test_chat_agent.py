#!/usr/bin/env python3
"""
Test WebSocket chat agent integration
"""
import asyncio
import json
import websockets

async def test_chat_agent():
    """Test chat agent through WebSocket"""
    uri = "ws://localhost:8000/ws/projects/demo-project"
    
    async with websockets.connect(uri) as websocket:
        print("✓ Connected to WebSocket")
        
        # Receive HELLO and STAGE_STATUS
        await websocket.recv()  # HELLO
        await websocket.recv()  # STAGE_STATUS
        print("✓ Received initial messages")
        
        # Test chat messages
        test_messages = [
            "What is the current stage?",
            "Can you explain what happens during training?",
            "Please retrain the model with higher learning rate",
            "I want to export the model",
            "Can you use Random Forest instead of XGBoost?",
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n🧪 Test {i}: Sending '{message}'")
            
            # Send chat message
            await websocket.send(json.dumps({
                "type": "chat",
                "text": message
            }))
            
            # Receive responses
            received_count = 0
            timeout = 5.0  # 5 seconds timeout
            
            try:
                while received_count < 3:  # Expect 1-3 responses per message
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    data = json.loads(response)
                    
                    if data.get("event", {}).get("name") == "LOG_LINE":
                        payload = data["event"]["payload"]
                        text = payload.get("text", "")
                        
                        # Print agent responses
                        if "Assistant:" in text or "Suggested action:" in text:
                            print(f"  📨 {text}")
                            received_count += 1
                            
                            # If this is the last expected response, break
                            if "Suggested action:" in text or received_count >= 2:
                                break
                    
            except asyncio.TimeoutError:
                print(f"  ⚠️  Timeout waiting for response (received {received_count} messages)")
                break
        
        print("\n✅ Chat agent test completed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_chat_agent())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
