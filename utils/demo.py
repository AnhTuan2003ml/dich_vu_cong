import socketio  # type: ignore
import base64
import json

# Socket.IO client for Citizen Card Service
card_service_socket = socketio.Client()

# Event handlers for Citizen Card Service
@card_service_socket.event
def connect():
    print("Connected to Citizen Card Service.")

@card_service_socket.event
def disconnect():
    print("Disconnected from Citizen Card Service.")

@card_service_socket.on("/event")
def handle_card_event(data):
    event_id = data.get("id")

    if event_id == 1:  # Sự kiện có thẻ quét
        print("New card detected.")
    elif event_id == 2:  # Đọc thẻ thành công (dữ liệu text)
        print("Card read successfully!")
        card_data = data.get("data", {})
        print(json.dumps(card_data, indent=4, ensure_ascii=False))
    elif event_id == 4:  # Đọc thẻ thành công (dữ liệu ảnh)
        print("Card image received.")
        img_data = data["data"].get("img_data")
        if img_data:
            with open("card_image.jpg", "wb") as img_file:
                img_file.write(base64.b64decode(img_data))
            print("Card image saved as card_image.jpg.")
    elif event_id == 3:  # Đọc thẻ thất bại
        print("Failed to read card:", data.get("message"))
    else:
        print("Unknown event ID:", event_id)

# Connect to Citizen Card Service
try:
    print("Connecting to Citizen Card Service...")
    card_service_socket.connect("http://192.168.5.1:8000")

    print("Waiting for events. Press Ctrl+C to exit.")
    card_service_socket.wait()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    card_service_socket.disconnect()
