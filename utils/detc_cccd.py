import socketio  # type: ignore
import base64
import json
from utils.sound_util import speak
from choose_form import run_main  # Import form hoặc giao diện tương ứng

# Socket.IO client for Citizen Card Service
card_service_socket = socketio.Client()

# Helper function to save data to a file
def save_to_file(file_path, data, is_binary=False):
    try:
        with open(file_path, "wb" if is_binary else "w", encoding=None if is_binary else "utf-8") as file:
            if is_binary:
                file.write(base64.b64decode(data))
            else:
                json.dump(data, file, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving to file {file_path}: {e}")
        return False

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
        card_data = data.get("data", {})
        success = save_to_file("temp/card_data.json", card_data)
        if success:
            name = card_data.get("personName")
            greeting_message = f"Xin chào, {name}! Mời bạn chọn dịch vụ!"
            speak(greeting_message)
            run_main()
        else:
            print("Failed to save card data.")
    elif event_id == 4:  # Đọc thẻ thành công (dữ liệu ảnh)
        img_data = data["data"].get("img_data")
        if img_data:
            success = save_to_file("temp/card_image.jpg", img_data, is_binary=True)
    elif event_id == 3:  # Đọc thẻ thất bại
        print("Failed to read card:", data.get("message"))
    else:
        print("Unknown event ID:", event_id)

# Connect to Citizen Card Service
try:
    card_service_socket.connect("http://192.168.5.1:8000")
    card_service_socket.wait()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    card_service_socket.disconnect()
