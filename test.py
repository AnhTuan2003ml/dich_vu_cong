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

MODEL_PATH = "trained_model"
SERIAL_PORT = '/dev/ttyUSB0'  # Thay đổi port này tùy theo hệ thống của bạn
BAUD_RATE = 115200
card_service_socket = socketio.Client()
class MainApp:
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
            print("Vui lòng kiểm tra:")
            print("1. Địa chỉ IP của server thẻ (192.168.5.1)")
            print("2. Server thẻ đã được khởi động")
            print("3. Kết nối mạng giữa máy tính và server thẻ")
            speak("Không thể kết nối đến bộ đọc thẻ. Vui lòng kiểm tra kết nối mạng và khởi động lại chương trình.")

        # Khởi tạo camera với cơ chế thử lại
        self.camera = None
        self.camera_lock = threading.Lock()  # Thêm lock để đồng bộ hóa truy cập camera
        self.initialize_camera()

        # Load mô hình đã được huấn luyện
        try:
            # Load model từ đường dẫn đã lưu
            self.model = SentenceTransformer(MODEL_PATH)
            
            # Load encoded_templates
            encoded_templates_path = os.path.join(MODEL_PATH, "encoded_templates.npy")
            if os.path.exists(encoded_templates_path):
                # Load dữ liệu từ file .npy
                loaded_data = np.load(encoded_templates_path, allow_pickle=True).item()
                
                # Chuyển các giá trị từ list thành numpy array
                self.encoded_templates = {k: np.array(v) for k, v in loaded_data.items()}
                print("Đã tải encoded_templates thành công")
            else:
                print(f"Không tìm thấy file {encoded_templates_path}")
                # Tạo một encoded_templates trống để tránh lỗi
                self.encoded_templates = {}
        except Exception as e:
            print(f"Lỗi khi tải model hoặc encoded_templates: {e}")
            # Fallback: Nếu không load được, tạo model mới
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.encoded_templates = {}

        # Khởi tạo PyAudio
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

    def safe_camera_operation(self, operation):
        """Thực hiện thao tác với camera một cách an toàn"""
        with self.camera_lock:
            try:
                return operation()
            except Exception as e:
                print(f"Lỗi khi thao tác với camera: {e}")
                return None

    def initialize_camera(self, max_retries=3):
        """Khởi tạo camera với cơ chế thử lại"""
        def _initialize():
            # Thử giải phóng camera cũ nếu có
            if self.camera is not None:
                self.camera.release()
                time.sleep(0.5)

            # Thử mở camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Không thể mở camera")

            # Kiểm tra camera có hoạt động không
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise Exception("Không thể đọc frame từ camera")

            print("Đã khởi tạo camera thành công")
            return True

        for attempt in range(max_retries):
            try:
                if self.safe_camera_operation(_initialize):
                    return True
            except Exception as e:
                print(f"Lần thử {attempt + 1}/{max_retries} thất bại: {e}")
                time.sleep(0.5)

        print("Không thể khởi tạo camera sau nhiều lần thử")
        speak("Không thể kết nối với camera. Vui lòng kiểm tra lại thiết bị.")
        return False

    def handle_card_event(self, data):
        """Xử lý sự kiện từ thẻ"""
        event_id = data.get("id")
        if event_id == 2:  # Đọc thẻ thành công
            card_data = data.get("data", {})
            name = card_data.get("personName", "người dùng")
            id_cccd = card_data.get("idCode","")
            if (id_cccd):
                speak(f"Xin chào, {name}!")
                # Lưu dữ liệu vào file JSON trong thư mục temp
                os.makedirs("temp", exist_ok=True)
                with open("temp/card_data.json", "w", encoding="utf-8") as f:
                    json.dump(card_data, f, ensure_ascii=False, indent=4)
                
                # Lấy ảnh từ thẻ CCCD
                if os.path.exists("temp/card_image.jpg"):
                    speak("Đã nhận diện khuôn mặt từ thẻ CCCD!")
                    
                    # **CHỤP ẢNH KHUÔN MẶT MỚI TỪ CAMERA**
                    captured_face_path = self.capture_face()
                    if captured_face_path:
                        speak("Đã chụp ảnh khuôn mặt của bạn. Đang so sánh với ảnh thẻ...")
                        # So sánh khuôn mặt
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

    def record_audio(self, duration=3):
        """Ghi âm trực tiếp sử dụng PyAudio với điều chỉnh ngưỡng âm thanh động"""
        if not self.audio:
            return None

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RECORD_SECONDS = duration

        try:
            # Tìm thiết bị USB Audio và lấy tần số lấy mẫu phù hợp
            device_index, sample_rate = self.find_usb_audio_device()
            device_info = self.audio.get_device_info_by_index(device_index)
            print(f"Đang sử dụng thiết bị: {device_info['name']}")
            print(f"Tần số lấy mẫu: {sample_rate}Hz")

            # Tạo thư mục audio nếu chưa tồn tại
            os.makedirs("audio", exist_ok=True)

            stream = self.audio.open(format=FORMAT,
                                   channels=CHANNELS,
                                   rate=int(sample_rate),  # Sử dụng tần số lấy mẫu từ thiết bị
                                   input=True,
                                   input_device_index=device_index,
                                   frames_per_buffer=CHUNK)

            print("Đang ghi âm...")
            frames = []
            
            # Biến để theo dõi mức âm thanh
            max_amplitude = 0
            silence_threshold = 500  # Ngưỡng im lặng ban đầu
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
                        silence_threshold = max_amplitude * 0.1  # 10% của biên độ lớn nhất
                    
                    # Phát hiện giọng nói
                    if current_amplitude > silence_threshold:
                        is_speaking = True
                        silence_counter = 0
                    elif is_speaking:
                        silence_counter += 1
                        # Nếu im lặng quá lâu, kết thúc ghi âm sớm
                        if silence_counter > int(sample_rate / CHUNK * 1.0):  # 1 giây im lặng
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
            wf.setframerate(int(sample_rate))  # Sử dụng tần số lấy mẫu từ thiết bị
            wf.writeframes(b''.join(frames))
            wf.close()

            print(f"Đã lưu file âm thanh tại: {temp_file}")
            return temp_file
        except Exception as e:
            print(f"Lỗi khi ghi âm: {e}")
            return None

    def clear_cccd_info(self):
        """Xóa thông tin CCCD và reset trạng thái xác thực"""
        # Xóa file ảnh thẻ
        if os.path.exists("temp/card_image.jpg"):
            try:
                os.remove("temp/card_image.jpg")
                print("Đã xóa ảnh thẻ CCCD")
            except Exception as e:
                print(f"Lỗi khi xóa ảnh thẻ: {e}")

        # Xóa file dữ liệu thẻ
        if os.path.exists("temp/card_data.json"):
            try:
                os.remove("temp/card_data.json")
                print("Đã xóa dữ liệu thẻ CCCD")
            except Exception as e:
                print(f"Lỗi khi xóa dữ liệu thẻ: {e}")

        # Reset trạng thái xác thực
        self.is_authenticated = False
        self.current_user = None
        print("Đã reset trạng thái xác thực")

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
                            # Xóa thông tin CCCD và yêu cầu quét lại
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
        if not self.audio or not self.speech:
            speak("Lỗi: Không thể khởi tạo microphone")
            print("Không thể sử dụng microphone. Vui lòng kiểm tra lại thiết bị.")
            return

        speak("Tôi đang nghe")
        try:
            # Ghi âm sử dụng PyAudio
            audio_file = self.record_audio(duration=5)  # Giữ nguyên thời gian ghi âm 10 giây
            if not audio_file:
                speak("Không thể ghi âm. Vui lòng thử lại.")
                return

            # Nhận diện giọng nói từ file đã ghi
            with sr.AudioFile(audio_file) as source:
                # Tăng độ nhạy của recognizer
                self.speech.energy_threshold = 100  # Giữ nguyên ngưỡng năng lượng
                self.speech.dynamic_energy_threshold = True
                self.speech.pause_threshold = 0.3  # Giảm thời gian chờ giữa các từ
                self.speech.phrase_threshold = 0.3  # Giữ nguyên ngưỡng cụm từ
                self.speech.non_speaking_duration = 0.3  # Giảm thời gian không nói
                
                audio = self.speech.record(source)
                try:
                    text = self.speech.recognize_google(audio, language='vi-VN')
                    print(f"Bạn nói: {text}")
                    
                    # Thực hiện dự đoán với ngưỡng tin cậy
                    best_match, confidence = self.predict_action(text)
                    
                    # Ngưỡng độ tin cậy để quyết định có hỏi lại hay không
                    THRESHOLD = 0.4  # Giữ nguyên ngưỡng tin cậy
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
            # Xóa file âm thanh nếu có lỗi
            if 'audio_file' in locals() and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    print(f"Đã xóa file âm thanh sau khi gặp lỗi: {audio_file}")
                except Exception as e:
                    print(f"Lỗi khi xóa file âm thanh: {e}")

    def predict_action(self, input_text):
        """Dự đoán hành động dựa trên văn bản đầu vào và trả về hành động tốt nhất cũng như độ tin cậy"""
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

    def ask_for_confirmation(self, original_text, suggested_action):
        """Hiển thị xác nhận với người dùng"""
        confirmation_text = f"Bạn muốn {suggested_action} phải không?"
        speak(confirmation_text)
        print(confirmation_text)
        print("Mời bạn chọn dịch vụ")

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

    def exit_app(self):
        speak("Rất vui được phục vụ bạn, hẹn gặp lại!")
        file_path = "greeting.mp3"
        if os.path.exists(file_path):
            os.remove(file_path)
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        # Reset trạng thái xác thực khi thoát
        self.is_authenticated = False
        self.current_user = None
        print("Tạm biệt!")

    def send_serial_response(self, message):
        """Gửi phản hồi qua cổng serial"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(f"{message}\n".encode('utf-8'))
                print(f"Đã gửi: {message}")
            except Exception as e:
                print(f"Lỗi gửi serial: {e}")
        else:
            print("Không thể gửi phản hồi: Cổng serial không mở")

    def initialize_gui(self):
        """Khởi tạo giao diện hiển thị"""
        try:
            # Đóng tất cả cửa sổ hiện có
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Đợi một chút để đảm bảo cửa sổ đã đóng
            
            # Tạo cửa sổ mới
            cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera Preview', 640, 480)
            
            # Đặt vị trí cửa sổ
            cv2.moveWindow('Camera Preview', 100, 100)
            
            return True
        except Exception as e:
            print(f"Lỗi khởi tạo giao diện: {e}")
            return False

    def refresh_camera(self):
        """Làm mới camera và giao diện"""
        def _refresh():
            # Đóng camera hiện tại
            if self.camera:
                self.camera.release()
                time.sleep(0.5)
            
            # Đóng tất cả cửa sổ
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            # Khởi tạo lại camera
            return self.initialize_camera()

        return self.safe_camera_operation(_refresh)

    def show_camera_preview(self):
        """Chụp ảnh khuôn mặt từ camera"""
        def _capture_face():
            if not self.camera:
                print("Camera không khả dụng")
                return None

            countdown_started = False
            countdown_value = 3
            last_countdown_time = 0
            face_detected = False
            face_image = None

            try:
                while True:
                    ret, frame = self.camera.read()
                    if not ret:
                        print("Không thể đọc frame từ camera")
                        break

                    # Chuyển đổi từ BGR sang RGB để phát hiện khuôn mặt
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)

                    if face_locations:
                        # Lấy khuôn mặt đầu tiên tìm thấy
                        top, right, bottom, left = face_locations[0]
                        face_detected = True

                        # Bắt đầu đếm ngược khi phát hiện khuôn mặt
                        current_time = time.time()
                        if not countdown_started:
                            countdown_started = True
                            last_countdown_time = current_time
                            countdown_value = 3
                            speak(str(countdown_value))

                        # Xử lý đếm ngược
                        if countdown_started:
                            elapsed_time = current_time - last_countdown_time
                            if elapsed_time >= 1:
                                countdown_value -= 1
                                last_countdown_time = current_time
                                if countdown_value > 0:
                                    speak(str(countdown_value))
                            
                            if countdown_value <= 0:
                                # Chụp ảnh khi đếm ngược kết thúc
                                face_image = frame[top:bottom, left:right]
                                break
                    else:
                        countdown_started = False
                        face_detected = False

            except Exception as e:
                print(f"Lỗi khi chụp ảnh: {e}")
                return None

            if face_image is not None:
                os.makedirs("temp", exist_ok=True)
                face_path = "temp/captured_face.jpg"
                cv2.imwrite(face_path, face_image)
                return face_path
            return None

        return self.safe_camera_operation(_capture_face)

    def capture_face_video(self, duration=10):
        """Chụp video khuôn mặt trong một khoảng thời gian"""
        def _capture_video():
            if not self.camera:
                print("Camera không khả dụng")
                return None

            # Lưu trữ các frame có khuôn mặt
            face_frames = []
            
            # Đếm ngược 3 giây trước khi bắt đầu quay
            speak("Chuẩn bị")
            countdown_value = 3
            while countdown_value > 0:
                speak(str(countdown_value))
                time.sleep(1)
                countdown_value -= 1
            
            speak("Bắt đầu")
            start_time = time.time()
            
            while time.time() - start_time < duration:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("Lỗi đọc frame từ camera, đang thử lại...")
                    continue

                # Chuyển đổi từ BGR sang RGB để phát hiện khuôn mặt
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                if face_locations:
                    # Lấy khuôn mặt đầu tiên tìm thấy
                    top, right, bottom, left = face_locations[0]
                    face_frames.append(frame[top:bottom, left:right])

            if not face_frames:
                print("Không tìm thấy khuôn mặt trong video")
                return None

            # Lưu video khuôn mặt
            os.makedirs("temp", exist_ok=True)
            video_path = "temp/captured_face_video.mp4"
            
            try:
                # Lấy kích thước của frame đầu tiên
                height, width = face_frames[0].shape[:2]
                
                # Tạo video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                
                # Ghi các frame vào video
                for frame in face_frames:
                    out.write(frame)
                
                out.release()
                print(f"Đã lưu video thành công tại: {video_path}")
                return video_path
            except Exception as e:
                print(f"Lỗi khi lưu video: {e}")
                return None

        return self.safe_camera_operation(_capture_video)

    def compare_faces(self, card_image_path, captured_video_path):
        """So sánh khuôn mặt từ ảnh thẻ và video chụp"""
        try:
            # Load ảnh thẻ
            card_image = face_recognition.load_image_file(card_image_path)
            card_face_locations = face_recognition.face_locations(card_image)
            
            if not card_face_locations:
                print("Không tìm thấy khuôn mặt trong ảnh thẻ")
                 # Làm mới camera nếu không tìm thấy khuôn mặt
                return False

            # Mã hóa khuôn mặt từ ảnh thẻ
            card_face_encoding = face_recognition.face_encodings(card_image, card_face_locations)[0]

            # Đọc video
            cap = cv2.VideoCapture(captured_video_path)
            if not cap.isOpened():
                print("Không thể mở video")
                 # Làm mới camera nếu không mở được video
                return False

            # Lưu trữ các độ tương đồng
            similarities = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Chuyển đổi từ BGR sang RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Tìm khuôn mặt trong frame
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    # Mã hóa khuôn mặt từ frame
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    
                    # Tính khoảng cách
                    face_distance = face_recognition.face_distance([card_face_encoding], face_encoding)[0]
                    similarity = 1 - face_distance
                    similarities.append(similarity)

            cap.release()

            if not similarities:
                print("Không tìm thấy khuôn mặt trong video")
                 # Làm mới camera nếu không tìm thấy khuôn mặt
                return False

            # Tính độ tương đồng trung bình
            avg_similarity = sum(similarities) / len(similarities)
            print(f"Độ tương đồng trung bình: {avg_similarity:.2%}")
            
            # Ngưỡng độ tương đồng
            SIMILARITY_THRESHOLD = 0.5
            
            # Kiểm tra độ tương đồng
            if avg_similarity >= SIMILARITY_THRESHOLD:
                print("Xác thực khuôn mặt thành công")
                # Xóa file video sau khi xác thực
                if os.path.exists(captured_video_path):
                    os.remove(captured_video_path)
                return True
            else:
                print("Xác thực khuôn mặt thất bại")
                # Xóa file video sau khi xác thực thất bại
                if os.path.exists(captured_video_path):
                    os.remove(captured_video_path)
                return False

        except Exception as e:
            print(f"Lỗi khi so sánh khuôn mặt: {e}")
            # Xóa file video nếu có lỗi
            if os.path.exists(captured_video_path):
                os.remove(captured_video_path)
            return False

    def capture_face(self):
        """Chụp video khuôn mặt từ camera"""
        if not self.camera:
            print("Camera không khả dụng")
            return None

        speak("Vui lòng nhìn thẳng vào camera")
        return self.capture_face_video(duration=7)

    def __del__(self):
        """Dọn dẹp khi đóng chương trình"""
        with self.camera_lock:
            if hasattr(self, 'camera') and self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
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