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
import torch
import torchaudio
from sentence_transformers import SentenceTransformer
from utils.sound_util import speak
from WinForm.giay_tam_tru import run_gtt
import threading
import queue
import sounddevice as sd

# Kiểm tra và sử dụng CUDA nếu có
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {DEVICE}")

MODEL_PATH = "trained_model"
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
card_service_socket = socketio.Client()

class JetsonApp:
    def __init__(self):
        # Khởi tạo kết nối serial
        try:
            self.serial_port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Đã kết nối với cổng serial {SERIAL_PORT}")
        except Exception as e:
            print(f"Lỗi kết nối serial: {e}")
            self.serial_port = None

        # Thêm biến theo dõi trạng thái xác thực
        self.is_authenticated = False
        self.current_user = None

        # Kết nối socket với xử lý lỗi
        try:
            print("Đang kết nối đến server thẻ...")
            card_service_socket.connect("http://192.168.5.1:8000", wait_timeout=10)
            card_service_socket.on("/event", self.handle_card_event)
            print("Đã kết nối thành công đến server thẻ")
        except Exception as e:
            print(f"Không thể kết nối đến server thẻ: {e}")
            speak("Không thể kết nối đến bộ đọc thẻ. Vui lòng kiểm tra kết nối mạng và khởi động lại chương trình.")

        # Khởi tạo camera với CUDA acceleration
        self.camera = None
        self.camera_lock = threading.Lock()
        self.initialize_camera()

        # Load mô hình đã được huấn luyện với CUDA
        try:
            self.model = SentenceTransformer(MODEL_PATH).to(DEVICE)
            encoded_templates_path = os.path.join(MODEL_PATH, "encoded_templates.npy")
            if os.path.exists(encoded_templates_path):
                loaded_data = np.load(encoded_templates_path, allow_pickle=True).item()
                self.encoded_templates = {k: torch.tensor(v).to(DEVICE) for k, v in loaded_data.items()}
                print("Đã tải encoded_templates thành công")
            else:
                print(f"Không tìm thấy file {encoded_templates_path}")
                self.encoded_templates = {}
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(DEVICE)
            self.encoded_templates = {}

        # Khởi tạo PyAudio với CUDA acceleration
        try:
            self.audio = pyaudio.PyAudio()
            self.speech = sr.Recognizer()
            self.audio_queue = queue.Queue()
            print("Đã khởi tạo PyAudio thành công")
        except Exception as e:
            print(f"Lỗi khởi tạo PyAudio: {e}")
            self.audio = None
            self.speech = None

        self.actions = [
            "tra cứu bảo hiểm", "cấp lại bằng lái xe", 
            "làm giấy tạm trú", "đăng ký hộ khẩu", 
            "cấp đổi căn cước công dân", "đăng ký kết hôn",
            "khai sinh cho trẻ em", "chứng thực giấy tờ"
        ]

    def initialize_camera(self):
        """Khởi tạo camera với CUDA acceleration"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Không thể mở camera")

            # Cấu hình camera cho CUDA
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            # Kiểm tra camera
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Không thể đọc frame từ camera")

            print("Đã khởi tạo camera thành công với CUDA acceleration")
            return True
        except Exception as e:
            print(f"Lỗi khởi tạo camera: {e}")
            return False

    def record_audio(self, duration=3):
        """Ghi âm với CUDA acceleration"""
        if not self.audio:
            return None

        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RECORD_SECONDS = duration

        try:
            # Tìm thiết bị USB Audio
            device_index, sample_rate = self.find_usb_audio_device()
            device_info = self.audio.get_device_info_by_index(device_index)
            print(f"Đang sử dụng thiết bị: {device_info['name']}")
            print(f"Tần số lấy mẫu: {sample_rate}Hz")

            # Tạo thư mục audio
            os.makedirs("audio", exist_ok=True)

            # Khởi tạo stream với CUDA acceleration
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=int(sample_rate),
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK
            )

            print("Đang ghi âm...")
            frames = []
            
            # Sử dụng CUDA cho xử lý âm thanh
            audio_buffer = torch.zeros((int(sample_rate * RECORD_SECONDS),), device=DEVICE)
            buffer_index = 0
            
            for i in range(0, int(sample_rate / CHUNK * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Chuyển đổi dữ liệu âm thanh sang tensor và đưa lên GPU
                    audio_data = torch.frombuffer(data, dtype=torch.float32).to(DEVICE)
                    audio_buffer[buffer_index:buffer_index + len(audio_data)] = audio_data
                    buffer_index += len(audio_data)
                    
                    # Tính toán mức âm thanh trên GPU
                    current_amplitude = torch.abs(audio_data).mean().item()
                    
                    # Phát hiện giọng nói
                    if current_amplitude > 0.1:  # Ngưỡng có thể điều chỉnh
                        silence_counter = 0
                    else:
                        silence_counter += 1
                        if silence_counter > int(sample_rate / CHUNK * 1.0):
                            break
                    
                except Exception as e:
                    print(f"Lỗi khi đọc chunk {i}: {e}")
                    continue

            print("Đã ghi âm xong")
            stream.stop_stream()
            stream.close()

            # Tạo tên file với timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            temp_file = f"audio/recording_{timestamp}.wav"
            
            # Lưu file với chất lượng cao
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(int(sample_rate))
            wf.writeframes(b''.join(frames))
            wf.close()

            print(f"Đã lưu file âm thanh tại: {temp_file}")
            return temp_file
        except Exception as e:
            print(f"Lỗi khi ghi âm: {e}")
            return None

    def compare_faces(self, card_image_path, captured_video_path):
        """So sánh khuôn mặt với CUDA acceleration"""
        try:
            # Load và chuyển ảnh thẻ lên GPU
            card_image = face_recognition.load_image_file(card_image_path)
            card_face_locations = face_recognition.face_locations(card_image, model="cnn")
            
            if not card_face_locations:
                print("Không tìm thấy khuôn mặt trong ảnh thẻ")
                return False

            # Mã hóa khuôn mặt từ ảnh thẻ
            card_face_encoding = face_recognition.face_encodings(card_image, card_face_locations)[0]
            card_face_encoding = torch.tensor(card_face_encoding).to(DEVICE)

            # Đọc video
            cap = cv2.VideoCapture(captured_video_path)
            if not cap.isOpened():
                print("Không thể mở video")
                return False

            # Lưu trữ các độ tương đồng
            similarities = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Chuyển đổi frame sang RGB và đưa lên GPU
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
                
                if face_locations:
                    # Mã hóa khuôn mặt từ frame
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    face_encoding = torch.tensor(face_encoding).to(DEVICE)
                    
                    # Tính khoảng cách trên GPU
                    face_distance = torch.norm(card_face_encoding - face_encoding)
                    similarity = 1 - face_distance.item()
                    similarities.append(similarity)

            cap.release()

            if not similarities:
                print("Không tìm thấy khuôn mặt trong video")
                return False

            # Tính độ tương đồng trung bình
            avg_similarity = sum(similarities) / len(similarities)
            print(f"Độ tương đồng trung bình: {avg_similarity:.2%}")
            
            # Ngưỡng độ tương đồng
            SIMILARITY_THRESHOLD = 0.5
            
            # Kiểm tra độ tương đồng
            if avg_similarity >= SIMILARITY_THRESHOLD:
                print("Xác thực khuôn mặt thành công")
                if os.path.exists(captured_video_path):
                    os.remove(captured_video_path)
                return True
            else:
                print("Xác thực khuôn mặt thất bại")
                if os.path.exists(captured_video_path):
                    os.remove(captured_video_path)
                return False

        except Exception as e:
            print(f"Lỗi khi so sánh khuôn mặt: {e}")
            if os.path.exists(captured_video_path):
                os.remove(captured_video_path)
            return False

    def predict_action(self, input_text):
        """Dự đoán hành động với CUDA acceleration"""
        # Chuyển input text lên GPU
        input_embedding = self.model.encode([input_text])[0]
        input_embedding = torch.tensor(input_embedding).to(DEVICE)
        
        best_score = -float("inf")
        best_action = None

        for action, embeddings in self.encoded_templates.items():
            # Tính toán độ tương đồng trên GPU
            similarities = torch.matmul(embeddings, input_embedding) / (
                torch.norm(embeddings, dim=1) * torch.norm(input_embedding)
            )
            max_similarity = torch.max(similarities).item()

            if max_similarity > best_score:
                best_score = max_similarity
                best_action = action

        return best_action, best_score

    def __del__(self):
        """Dọn dẹp khi đóng chương trình"""
        with self.camera_lock:
            if hasattr(self, 'camera') and self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        # Giải phóng bộ nhớ GPU
        torch.cuda.empty_cache()

def run_app():
    app = JetsonApp()
    print("\nDanh sách dịch vụ có sẵn:")
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