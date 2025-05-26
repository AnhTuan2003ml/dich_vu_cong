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
            print("Đã khởi tạo camera Jetson thành công")
        except Exception as e:
            print(f"Lỗi khởi tạo camera Jetson: {e}")
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
            print(f"Lỗi trong on_new_sample: {e}")
        return Gst.FlowReturn.ERROR

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            self.is_running = False
            err, debug = message.parse_error()
            print(f"Lỗi: {err.message}")
            print(f"Thông tin debug: {debug}")

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
        # Khởi tạo CUDA
        try:
            cuda_runtime.init()
            print("Đã khởi tạo CUDA thành công")
        except Exception as e:
            print(f"Lỗi khởi tạo CUDA: {e}")

        # Khởi tạo kết nối serial
        try:
            self.serial_port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Đã kết nối với cổng serial {SERIAL_PORT}")
        except Exception as e:
            print(f"Lỗi kết nối serial: {e}")
            self.serial_port = None

        # Trạng thái xác thực
        self.is_authenticated = False
        self.current_user = None

        # Kết nối với server thẻ
        try:
            print("Đang kết nối đến server thẻ...")
            card_service_socket.connect("http://192.168.5.1:8000", wait_timeout=10)
            card_service_socket.on("/event", self.handle_card_event)
            print("Đã kết nối thành công đến server thẻ")
        except Exception as e:
            print(f"Không thể kết nối đến server thẻ: {e}")
            print("Vui lòng kiểm tra:")
            print("1. Địa chỉ IP của server thẻ (192.168.5.1)")
            print("2. Server thẻ đã được khởi động")
            print("3. Kết nối mạng giữa máy tính và server thẻ")
            speak("Không thể kết nối đến bộ đọc thẻ. Vui lòng kiểm tra kết nối mạng và khởi động lại chương trình.")

        # Khởi tạo camera Jetson
        self.camera = JetsonCamera()
        if not self.camera.is_running:
            speak("Không thể khởi tạo camera. Vui lòng kiểm tra thiết bị.")

        # Load model
        try:
            self.model = SentenceTransformer(MODEL_PATH)
            encoded_templates_path = os.path.join(MODEL_PATH, "encoded_templates.npy")
            if os.path.exists(encoded_templates_path):
                loaded_data = np.load(encoded_templates_path, allow_pickle=True).item()
                self.encoded_templates = {k: np.array(v) for k, v in loaded_data.items()}
                print("Đã tải encoded_templates thành công")
            else:
                print(f"Không tìm thấy file {encoded_templates_path}")
                self.encoded_templates = {}
        except Exception as e:
            print(f"Lỗi khi tải model hoặc encoded_templates: {e}")
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.encoded_templates = {}

        # Khởi tạo audio
        try:
            self.audio = pyaudio.PyAudio()
            self.speech = sr.Recognizer()
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

    def handle_card_event(self, data):
        """Xử lý sự kiện từ thẻ"""
        event_id = data.get("id")
        if event_id == 2:  # Đọc thẻ thành công
            card_data = data.get("data", {})
            name = card_data.get("personName", "người dùng")
            id_cccd = card_data.get("idCode","")
            if (id_cccd):
                speak(f"Xin chào, {name}!")
                os.makedirs("temp", exist_ok=True)
                with open("temp/card_data.json", "w", encoding="utf-8") as f:
                    json.dump(card_data, f, ensure_ascii=False, indent=4)
                
                if os.path.exists("temp/card_image.jpg"):
                    speak("Đã nhận diện khuôn mặt từ thẻ CCCD!")
                    captured_face_path = self.capture_face()
                    if captured_face_path:
                        speak("Đã chụp ảnh khuôn mặt của bạn. Đang so sánh với ảnh thẻ...")
                        matched = self.compare_faces("temp/card_image.jpg", captured_face_path)
                        if matched:
                            speak("Khuôn mặt xác thực thành công!")
                            self.is_authenticated = True
                            self.current_user = name
                        else:
                            speak("Khuôn mặt không khớp với ảnh trên thẻ. Vui lòng thử lại.")
                            self.is_authenticated = False
                            self.current_user = None
                    else:
                        speak("Không thể chụp ảnh khuôn mặt. Vui lòng kiểm tra camera.")
                        self.is_authenticated = False
                        self.current_user = None
        elif event_id == 4:  # Nhận ảnh từ thẻ
            img_data = data.get("data", {}).get("img_data")
            if img_data:
                os.makedirs("temp", exist_ok=True)
                with open("temp/card_image.jpg", "wb") as img_file:
                    if isinstance(img_data, str):
                        import base64
                        img_file.write(base64.b64decode(img_data))
                    elif isinstance(img_data, bytes):
                        img_file.write(img_data)
                print("Đã lưu ảnh từ thẻ CCCD")

    def capture_face(self):
        """Chụp ảnh khuôn mặt sử dụng camera Jetson"""
        if not self.camera.is_running:
            print("Camera không khả dụng")
            return None

        speak("Vui lòng nhìn thẳng vào camera")
        face_frames = []
        start_time = time.time()
        
        while time.time() - start_time < 7:  # Chụp trong 7 giây
            ret, frame = self.camera.read()
            if not ret or frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_frames.append(frame[top:bottom, left:right])

        if not face_frames:
            print("Không tìm thấy khuôn mặt trong video")
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
            print(f"Đã lưu video thành công tại: {video_path}")
            return video_path
        except Exception as e:
            print(f"Lỗi khi lưu video: {e}")
            return None

    def compare_faces(self, card_image_path, captured_video_path):
        """So sánh khuôn mặt sử dụng GPU acceleration"""
        try:
            card_image = face_recognition.load_image_file(card_image_path)
            card_face_locations = face_recognition.face_locations(card_image)
            
            if not card_face_locations:
                print("Không tìm thấy khuôn mặt trong ảnh thẻ")
                return False

            card_face_encoding = face_recognition.face_encodings(card_image, card_face_locations)[0]

            cap = cv2.VideoCapture(captured_video_path)
            if not cap.isOpened():
                print("Không thể mở video")
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
                print("Không tìm thấy khuôn mặt trong video")
                return False

            avg_similarity = sum(similarities) / len(similarities)
            print(f"Độ tương đồng trung bình: {avg_similarity:.2%}")
            
            SIMILARITY_THRESHOLD = 0.5
            
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

    def read_serial_command(self):
        """Đọc lệnh từ cổng serial"""
        if self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting:
                    data = self.serial_port.readline()
                    # Kiểm tra nếu dữ liệu là hex
                    if len(data) >= 9 and data[0] == 0x5A and data[1] == 0xA5:
                        # Lấy mã từ byte cuối cùng
                        code = data[-1]
                        if code == 0x01:
                            speak("Mời bạn chọn dịch vụ")
                            return "CHOOSE_SERVICE"
                        elif code == 0x02:
                            speak("Xin chào, hẹn gặp lại")
                            self.clear_cccd_info()
                            return "GOODBYE"
                        elif code == 0x03:
                            return "START_LISTENING"
                    # Nếu không phải hex, xử lý như text thông thường
                    command = data.decode('utf-8').strip()
                    return command
            except Exception as e:
                print(f"Lỗi đọc serial: {e}")
        return None

    def start_listening(self):
        """Bắt đầu lắng nghe giọng nói"""
        if not self.audio or not self.speech:
            speak("Lỗi: Không thể khởi tạo microphone")
            print("Không thể sử dụng microphone. Vui lòng kiểm tra lại thiết bị.")
            return

        speak("Tôi đang nghe")
        try:
            # Ghi âm sử dụng PyAudio
            audio_file = self.record_audio(duration=5)
            if not audio_file:
                speak("Không thể ghi âm. Vui lòng thử lại.")
                return

            # Nhận diện giọng nói từ file đã ghi
            with sr.AudioFile(audio_file) as source:
                self.speech.energy_threshold = 100
                self.speech.dynamic_energy_threshold = True
                self.speech.pause_threshold = 0.3
                self.speech.phrase_threshold = 0.3
                self.speech.non_speaking_duration = 0.3
                
                audio = self.speech.record(source)
                try:
                    text = self.speech.recognize_google(audio, language='vi-VN')
                    print(f"Bạn nói: {text}")
                    
                    # Thực hiện dự đoán với ngưỡng tin cậy
                    best_match, confidence = self.predict_action(text)
                    
                    # Ngưỡng độ tin cậy để quyết định có hỏi lại hay không
                    THRESHOLD = 0.4
                    if best_match:
                        # Thực hiện hành động ngay nếu có kết quả dự đoán
                        self.perform_action(best_match)
                    else:
                        speak("Xin lỗi, tôi không hiểu yêu cầu của bạn. Vui lòng thử lại.")
                        print("Mời bạn chọn dịch vụ")
                    
                except sr.UnknownValueError:
                    print("Không nghe rõ. Vui lòng nói lại")
                    speak("Không nghe rõ. Vui lòng nói lại")
                except sr.RequestError as e:
                    print(f"Lỗi kết nối với Google Speech Recognition: {e}")
                    speak("Lỗi kết nối. Vui lòng thử lại.")

            # Xóa file âm thanh sau khi xử lý xong
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    print(f"Đã xóa file âm thanh: {audio_file}")
                except Exception as e:
                    print(f"Lỗi khi xóa file âm thanh: {e}")

        except Exception as e:
            print(f"Lỗi khi sử dụng microphone: {e}")
            if 'audio_file' in locals() and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    print(f"Đã xóa file âm thanh sau khi gặp lỗi: {audio_file}")
                except Exception as e:
                    print(f"Lỗi khi xóa file âm thanh: {e}")

    def record_audio(self, duration=3):
        """Ghi âm trực tiếp sử dụng PyAudio"""
        if not self.audio:
            return None

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RECORD_SECONDS = duration

        try:
            # Tìm thiết bị USB Audio
            device_index, sample_rate = self.find_usb_audio_device()
            device_info = self.audio.get_device_info_by_index(device_index)
            print(f"Đang sử dụng thiết bị: {device_info['name']}")
            print(f"Tần số lấy mẫu: {sample_rate}Hz")

            # Tạo thư mục audio nếu chưa tồn tại
            os.makedirs("audio", exist_ok=True)

            stream = self.audio.open(format=FORMAT,
                                   channels=CHANNELS,
                                   rate=int(sample_rate),
                                   input=True,
                                   input_device_index=device_index,
                                   frames_per_buffer=CHUNK)

            print("Đang ghi âm...")
            frames = []
            
            # Biến để theo dõi mức âm thanh
            max_amplitude = 0
            silence_threshold = 500
            silence_counter = 0
            is_speaking = False
            
            # Ghi âm theo từng chunk
            for i in range(0, int(sample_rate / CHUNK * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Tính toán mức âm thanh hiện tại
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    current_amplitude = np.abs(audio_data).mean()
                    
                    # Cập nhật biên độ lớn nhất
                    max_amplitude = max(max_amplitude, current_amplitude)
                    
                    # Điều chỉnh ngưỡng im lặng dựa trên biên độ lớn nhất
                    if max_amplitude > 0:
                        silence_threshold = max_amplitude * 0.1
                    
                    # Phát hiện giọng nói
                    if current_amplitude > silence_threshold:
                        is_speaking = True
                        silence_counter = 0
                    elif is_speaking:
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

    def find_usb_audio_device(self):
        """Tìm thiết bị USB Audio"""
        try:
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if 'USB' in device_info['name'] and device_info['maxInputChannels'] > 0:
                    print(f"Tìm thấy thiết bị USB Audio: {device_info['name']}")
                    return i, device_info['defaultSampleRate']
            print("Không tìm thấy thiết bị USB Audio, sử dụng thiết bị mặc định")
            default_device = self.audio.get_default_input_device_info()
            return default_device['index'], default_device['defaultSampleRate']
        except Exception as e:
            print(f"Lỗi khi tìm thiết bị USB Audio: {e}")
            default_device = self.audio.get_default_input_device_info()
            return default_device['index'], default_device['defaultSampleRate']

    def predict_action(self, input_text):
        """Dự đoán hành động dựa trên văn bản đầu vào"""
        input_embedding = self.model.encode([input_text])[0]
        best_score = -float("inf")
        best_action = None

        for action, embeddings in self.encoded_templates.items():
            similarities = np.dot(embeddings, input_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding))
            max_similarity = np.max(similarities)

            if max_similarity > best_score:
                best_score = max_similarity
                best_action = action

        return best_action, best_score

    def perform_action(self, command):
        """Thực hiện hành động dựa trên lệnh"""
        if not self.is_authenticated:
            speak("Vui lòng quét thẻ và xác thực khuôn mặt trước khi sử dụng dịch vụ.")
            return

        try:
            if command == "tra cứu bảo hiểm":
                speak("Bạn muốn tra cứu bảo hiểm gì?")
            elif command == "cấp lại bằng lái xe":
                speak("Mời bạn điền vào form sau!")
                from WinForm.cap_lai_bang_lai_xe import run
                run()
                speak("Bản in đang được tạo. Vui lòng chờ...")
            elif command == "làm giấy tạm trú":
                speak("Bạn điền thông tin vào phiếu khai sau. Sau đó mang đến quầy số 6")
                run_gtt()
                speak("Bản in đang được tạo. Vui lòng chờ...")
            elif command == "đăng ký hộ khẩu":
                speak("Tôi sẽ hướng dẫn bạn đăng ký hộ khẩu!")
            elif command == "cấp đổi căn cước công dân":
                speak("Tôi sẽ hướng dẫn bạn cấp đổi căn cước công dân!")
            elif command == "đăng ký kết hôn":
                speak("Tôi sẽ hướng dẫn bạn đăng ký kết hôn!")
            elif command == "khai sinh cho trẻ em":
                speak("Tôi sẽ hướng dẫn bạn làm giấy khai sinh cho trẻ em!")
            elif command == "chứng thực giấy tờ":
                speak("Tôi sẽ giúp bạn chứng thực giấy tờ!")
            else:
                speak("Xin lỗi, tôi không hiểu yêu cầu của bạn. Vui lòng thử lại.")
                print("Mời bạn chọn dịch vụ")
        except Exception as e:
            print(f"Lỗi khi thực hiện hành động: {e}")

    def clear_cccd_info(self):
        """Xóa thông tin CCCD và reset trạng thái xác thực"""
        if os.path.exists("temp/card_image.jpg"):
            try:
                os.remove("temp/card_image.jpg")
                print("Đã xóa ảnh thẻ CCCD")
            except Exception as e:
                print(f"Lỗi khi xóa ảnh thẻ: {e}")

        if os.path.exists("temp/card_data.json"):
            try:
                os.remove("temp/card_data.json")
                print("Đã xóa dữ liệu thẻ CCCD")
            except Exception as e:
                print(f"Lỗi khi xóa dữ liệu thẻ: {e}")

        self.is_authenticated = False
        self.current_user = None
        print("Đã reset trạng thái xác thực")

    def exit_app(self):
        """Thoát ứng dụng"""
        speak("Rất vui được phục vụ bạn, hẹn gặp lại!")
        file_path = "greeting.mp3"
        if os.path.exists(file_path):
            os.remove(file_path)
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.is_authenticated = False
        self.current_user = None
        print("Tạm biệt!")

    def __del__(self):
        """Dọn dẹp khi đóng chương trình"""
        if hasattr(self, 'camera'):
            self.camera.release()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()

def run_app():
    app = MainApp()
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