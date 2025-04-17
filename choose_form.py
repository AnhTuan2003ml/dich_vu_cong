import os
import json
import numpy as np
import speech_recognition as sr
import tkinter as tk
import tkinter.font as font
import socketio
from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer
from utils.sound_util import speak  # Import từ file hỗ trợ
from test_user.check_account import check_login,check_id_exists
from WinForm.Gd00 import create_login_window

MODEL_PATH = "trained_model"
card_service_socket = socketio.Client()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Ứng dụng dịch vụ')
        self.root.attributes('-fullscreen', True)

        self.custom_font = font.Font(family="Helvetica", size=16)

        # Cửa sổ chờ ban đầu
        self.waiting_window = tk.Toplevel(self.root)
        self.waiting_window.title("Quét thẻ để tiếp tục")
        self.waiting_window.geometry("400x300")

        self.image = Image.open("img\\bg_00.jpg")  # Đổi thành ảnh chờ bạn muốn
        self.image = self.image.resize((400, 250), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image)

        self.image_label = tk.Label(self.waiting_window, image=self.photo)
        self.image_label.pack()

        self.status_label = tk.Label(self.waiting_window, text="Vui lòng quét thẻ để tiếp tục", font=self.custom_font)
        self.status_label.pack(pady=10)

        self.root.withdraw()  # Ẩn cửa sổ chính đến khi quét thẻ

        card_service_socket.connect("http://192.168.5.1:8000")

        @card_service_socket.on("/event")
        def handle_card_event(data):
            event_id = data.get("id")
            if event_id == 2:  # Đọc thẻ thành công
                card_data = data.get("data", {})
                name = card_data.get("personName", "người dùng")
                id_cccd = card_data.get("idCode","")
                if check_id_exists(id_cccd):
                    speak(f"Xin chào, {name}! Mời bạn đăng nhập để thực hiện dịch vụ!")
                    # Lưu dữ liệu vào file JSON trong thư mục temp
                    os.makedirs("temp", exist_ok=True)  # Đảm bảo thư mục tồn tại
                    with open("temp/card_data.json", "w", encoding="utf-8") as f:
                        json.dump(card_data, f, ensure_ascii=False, indent=4)
                    create_login_window()
                    # self.waiting_window.destroy()
                    # self.root.deiconify()
                    # self.init_main_interface()
                else:
                    speak(f"Xin chào, {name}! Bạn chưa có tài khoản, mời bạn đăng kí!")
            elif event_id == 4:
                img_data = data.get("data", {}).get("img_data")
                if img_data:
                    os.makedirs("temp", exist_ok=True)  # Đảm bảo thư mục tồn tại
                    with open("temp/card_image.jpg", "wb") as img_file:
                        # Nếu dữ liệu ảnh là chuỗi base64, cần giải mã
                        if isinstance(img_data, str):
                            import base64
                            img_file.write(base64.b64decode(img_data))
                        # Nếu dữ liệu ảnh là bytes
                        elif isinstance(img_data, bytes):
                            img_file.write(img_data)

    def init_main_interface(self):
        # Xóa các nút cũ trước khi tạo lại các nút mới
        if hasattr(self, 'frame') and self.frame.winfo_exists():
            self.frame.destroy()  # Xóa frame cũ để tránh trùng lặp

        self.frame = tk.Frame(self.root, bg="#f0f0f0")
        self.frame.pack(expand=True, fill=tk.BOTH)

        self.label = tk.Label(self.frame, text="Chọn chức năng bạn muốn:", font=self.custom_font, bg="#f0f0f0")
        self.label.pack(pady=20)

        self.model = SentenceTransformer(MODEL_PATH)
        self.actions = self.load_actions("actions.json")
        self.encoded_templates = np.load(os.path.join(MODEL_PATH, "encoded_templates.npy"), allow_pickle=True).item()

        for action in self.actions.keys():
            self.create_button(action)

        self.bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.mic_button = tk.Button(self.bottom_frame, text="Bắt đầu nghe", command=self.start_listening,
                                    font=self.custom_font, bg="#4CAF50", fg="white")
        self.mic_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.exit_button = tk.Button(self.bottom_frame, text="Thoát", command=self.exit_app,
                                     font=self.custom_font, bg="#f44336", fg="white")
        self.exit_button.pack(side=tk.RIGHT, padx=20, pady=10)

        self.speech = sr.Recognizer()

    def load_actions(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def create_button(self, action):
        button = tk.Button(self.frame, text=action, command=lambda: self.perform_action(action),
                           font=self.custom_font, bg="#2196F3", fg="white")
        button.pack(pady=10, padx=20, fill=tk.X)

    def start_listening(self):
        speak("Tôi đang nghe")
        with sr.Microphone() as source:
            self.label.config(text="Đang nghe...")
            self.root.update()
            audio = self.speech.listen(source)
            try:
                text = self.speech.recognize_google(audio, language='vi-VN')
                self.label.config(text=f"Bạn nói: {text}")
                best_match = self.predict_action(text)
                self.perform_action(best_match)
            except sr.UnknownValueError:
                self.label.config(text="Mời bạn chọn dịch vụ")
            except sr.RequestError:
                self.label.config(text="Lỗi kết nối. Vui lòng thử lại.")

    def predict_action(self, input_text):
        input_embedding = self.model.encode([input_text])[0]
        best_score = -float("inf")
        best_action = None

        for action, embeddings in self.encoded_templates.items():
            similarities = np.dot(embeddings, input_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding))
            max_similarity = np.max(similarities)

            if max_similarity > best_score:
                best_score = max_similarity
                best_action = action

        return best_action

    def perform_action(self, command):
        from WinForm.cap_lai_bang_lai_xe import run  # Đảm bảo đường dẫn đúng
        if command == "tra cứu bảo hiểm":
            speak("Bạn muốn tra cứu bảo hiểm gì?")
        elif command == "cấp lại bằng lái xe":
            speak("Mời bạn điền vào form sau!")
            run()
        elif command == "làm giấy tạm trú":
            speak("Tôi sẽ hướng dẫn bạn điền vào đơn cấp giấy tạm trú!")

    def exit_app(self):
        # # Xóa thông tin của người hiện tại
        # if os.path.exists("temp/card_data.json"):
        #     os.remove("temp/card_data.json")  # Xóa file card data
        # if os.path.exists("temp/card_image.jpg"):
        #     os.remove("temp/card_image.jpg")  # Xóa file card data
        speak("Rất vui được phục vụ bạn, hẹn gặp lại!")
        
        # Ẩn cửa sổ chính
        self.root.withdraw()

        # Quay lại cửa sổ chờ (tạo lại nếu cần)
        self.reset_waiting_window()

    def reset_waiting_window(self):
        if not self.waiting_window.winfo_exists():
            # Nếu cửa sổ chờ đã bị đóng, tạo lại cửa sổ chờ
            self.waiting_window = tk.Toplevel(self.root)
            self.waiting_window.title("Quét thẻ để tiếp tục")
            self.waiting_window.geometry("400x300")
            
            self.image = Image.open("img\\bg_00.jpg")  # Đổi thành ảnh chờ bạn muốn
            self.image = self.image.resize((400, 250), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)
            
            self.image_label = tk.Label(self.waiting_window, image=self.photo)
            self.image_label.pack()
            
            self.status_label = tk.Label(self.waiting_window, text="Vui lòng quét thẻ để tiếp tục", font=self.custom_font)
            self.status_label.pack(pady=10)
        else:
            # Nếu cửa sổ chờ còn tồn tại, chỉ cần hiển thị lại
            self.waiting_window.deiconify()




if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
