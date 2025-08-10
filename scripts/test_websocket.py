import websocket
import json
import time

ws = websocket.WebSocket()
max_retries = 5
retry_delay = 2

for attempt in range(max_retries):
    try:
        ws.connect("ws://127.0.0.1:8765/mujoco")
        print("Connected to WebSocket server")
        ws.send(json.dumps({"test": "hello"}))
        print("Sent test message")
        response = ws.recv()
        print(f"Received: {response}")
        ws.close()
        break
    except Exception as e:
        print(f"Connection attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
else:
    print("Failed to connect to WebSocket server after retries")
