import os
import socketio
import json
import numpy as np
import speech_recognition as sr
import serial
import time
import pyaudio
import wave
import cv2
import face_recognition
from sentence_transformers import SentenceTransformer
from utils.sound_util import speak
from WinForm.giay_tam_tru import run_gtt
import threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import nvidia.cuda as cuda
import nvidia.cuda.runtime as cuda_runtime

# Initialize GStreamer
Gst.init(None)

MODEL_PATH = "trained_model"
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
card_service_socket = socketio.Client()

class JetsonCamera:
    def __init__(self):
        self.pipeline = None
        self.sink = None
        self.bus = None
        self.is_running = False
        self.frame = None
        self.lock = threading.Lock()
        self.initialize_pipeline()

    def initialize_pipeline(self):
        try:
            # Create GStreamer pipeline optimized for Jetson
            pipeline_str = (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
                "nvvidconv ! "
                "video/x-raw, format=BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! "
                "appsink name=sink"
            )
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.sink = self.pipeline.get_by_name("sink")
            self.sink.set_property("emit-signals", True)
            self.sink.connect("new-sample", self.on_new_sample)
            
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect("message", self.on_message)
            
            self.pipeline.set_state(Gst.State.PLAYING)
            self.is_running = True
            print("Jetson camera initialized successfully")
        except Exception as e:
            print(f"Error initializing Jetson camera: {e}")
            self.is_running = False

    def on_new_sample(self, sink):
        try:
            sample = sink.emit("pull-sample")
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()
                structure = caps.get_structure(0)
                width = structure.get_value("width")
                height = structure.get_value("height")
                
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    with self.lock:
                        self.frame = np.ndarray(
                            shape=(height, width, 3),
                            dtype=np.uint8,
                            buffer=map_info.data
                        ).copy()
                    buffer.unmap(map_info)
                return Gst.FlowReturn.OK
        except Exception as e:
            print(f"Error in on_new_sample: {e}")
        return Gst.FlowReturn.ERROR

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            self.is_running = False
            err, debug = message.parse_error()
            print(f"Error: {err.message}")
            print(f"Debug info: {debug}")

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
        return False, None

    def release(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False

class MainApp:
    def __init__(self):
        # Initialize CUDA
        try:
            cuda_runtime.init()
            print("CUDA initialized successfully")
        except Exception as e:
            print(f"Error initializing CUDA: {e}")

        # Initialize serial connection
        try:
            self.serial_port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Connected to serial port {SERIAL_PORT}")
        except Exception as e:
            print(f"Serial connection error: {e}")
            self.serial_port = None

        # Authentication state
        self.is_authenticated = False
        self.current_user = None

        # Connect to card service
        try:
            print("Connecting to card server...")
            card_service_socket.connect("http://192.168.5.1:8000", wait_timeout=10)
            card_service_socket.on("/event", self.handle_card_event)
            print("Successfully connected to card server")
        except Exception as e:
            print(f"Could not connect to card server: {e}")
            print("Please check:")
            print("1. Card server IP address (192.168.5.1)")
            print("2. Card server is running")
            print("3. Network connection between computer and card server")
            speak("Could not connect to card reader. Please check network connection and restart the program.")

        # Initialize Jetson camera
        self.camera = JetsonCamera()
        if not self.camera.is_running:
            speak("Could not initialize camera. Please check the device.")

        # Load model
        try:
            self.model = SentenceTransformer(MODEL_PATH)
            encoded_templates_path = os.path.join(MODEL_PATH, "encoded_templates.npy")
            if os.path.exists(encoded_templates_path):
                loaded_data = np.load(encoded_templates_path, allow_pickle=True).item()
                self.encoded_templates = {k: np.array(v) for k, v in loaded_data.items()}
                print("Successfully loaded encoded_templates")
            else:
                print(f"File {encoded_templates_path} not found")
                self.encoded_templates = {}
        except Exception as e:
            print(f"Error loading model or encoded_templates: {e}")
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.encoded_templates = {}

        # Initialize audio
        try:
            self.audio = pyaudio.PyAudio()
            self.speech = sr.Recognizer()
            print("Successfully initialized PyAudio")
        except Exception as e:
            print(f"Error initializing PyAudio: {e}")
            self.audio = None
            self.speech = None

        self.actions = [
            "tra cứu bảo hiểm", "cấp lại bằng lái xe", 
            "làm giấy tạm trú", "đăng ký hộ khẩu", 
            "cấp đổi căn cước công dân", "đăng ký kết hôn",
            "khai sinh cho trẻ em", "chứng thực giấy tờ"
        ]

    def handle_card_event(self, data):
        """Handle card events"""
        event_id = data.get("id")
        if event_id == 2:  # Successful card read
            card_data = data.get("data", {})
            name = card_data.get("personName", "user")
            id_cccd = card_data.get("idCode","")
            if (id_cccd):
                speak(f"Hello, {name}!")
                os.makedirs("temp", exist_ok=True)
                with open("temp/card_data.json", "w", encoding="utf-8") as f:
                    json.dump(card_data, f, ensure_ascii=False, indent=4)
                
                if os.path.exists("temp/card_image.jpg"):
                    speak("Face detected from ID card!")
                    captured_face_path = self.capture_face()
                    if captured_face_path:
                        speak("Face captured. Comparing with ID card...")
                        matched = self.compare_faces("temp/card_image.jpg", captured_face_path)
                        if matched:
                            speak("Face authentication successful!")
                            self.is_authenticated = True
                            self.current_user = name
                        else:
                            speak("Face does not match ID card. Please try again.")
                            self.is_authenticated = False
                            self.current_user = None
                    else:
                        speak("Could not capture face. Please check camera.")
                        self.is_authenticated = False
                        self.current_user = None
        elif event_id == 4:  # Receive image from card
            img_data = data.get("data", {}).get("img_data")
            if img_data:
                os.makedirs("temp", exist_ok=True)
                with open("temp/card_image.jpg", "wb") as img_file:
                    if isinstance(img_data, str):
                        import base64
                        img_file.write(base64.b64decode(img_data))
                    elif isinstance(img_data, bytes):
                        img_file.write(img_data)
                print("Saved ID card image")

    def capture_face(self):
        """Capture face using Jetson camera"""
        if not self.camera.is_running:
            print("Camera not available")
            return None

        speak("Please look at the camera")
        face_frames = []
        start_time = time.time()
        
        while time.time() - start_time < 7:  # 7 seconds capture
            ret, frame = self.camera.read()
            if not ret or frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_frames.append(frame[top:bottom, left:right])

        if not face_frames:
            print("No face detected in video")
            return None

        os.makedirs("temp", exist_ok=True)
        video_path = "temp/captured_face_video.mp4"
        
        try:
            height, width = face_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
            
            for frame in face_frames:
                out.write(frame)
            
            out.release()
            print(f"Video saved successfully at: {video_path}")
            return video_path
        except Exception as e:
            print(f"Error saving video: {e}")
            return None

    def compare_faces(self, card_image_path, captured_video_path):
        """Compare faces using GPU acceleration"""
        try:
            card_image = face_recognition.load_image_file(card_image_path)
            card_face_locations = face_recognition.face_locations(card_image)
            
            if not card_face_locations:
                print("No face found in ID card image")
                return False

            card_face_encoding = face_recognition.face_encodings(card_image, card_face_locations)[0]

            cap = cv2.VideoCapture(captured_video_path)
            if not cap.isOpened():
                print("Could not open video")
                return False

            similarities = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    face_distance = face_recognition.face_distance([card_face_encoding], face_encoding)[0]
                    similarity = 1 - face_distance
                    similarities.append(similarity)

            cap.release()

            if not similarities:
                print("No face found in video")
                return False

            avg_similarity = sum(similarities) / len(similarities)
            print(f"Average similarity: {avg_similarity:.2%}")
            
            SIMILARITY_THRESHOLD = 0.5
            
            if avg_similarity >= SIMILARITY_THRESHOLD:
                print("Face authentication successful")
                if os.path.exists(captured_video_path):
                    os.remove(captured_video_path)
                return True
            else:
                print("Face authentication failed")
                if os.path.exists(captured_video_path):
                    os.remove(captured_video_path)
                return False

        except Exception as e:
            print(f"Error comparing faces: {e}")
            if os.path.exists(captured_video_path):
                os.remove(captured_video_path)
            return False

    def __del__(self):
        """Cleanup when closing the program"""
        if hasattr(self, 'camera'):
            self.camera.release()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()

def run_app():
    app = MainApp()
    print("\nAvailable services:")
    for i, action in enumerate(app.actions, 1):
        print(f"{i}. {action}")
    
    while True:
        command = app.read_serial_command()
        if command:
            if command == "START_LISTENING":
                app.start_listening()
            elif command == "EXIT":
                app.exit_app()
                break
        time.sleep(0.1)

if __name__ == "__main__":
    run_app() 