import os
import json
import numpy as np
import speech_recognition as sr
import tkinter as tk
import tkinter.font as font
from tkinter import messagebox
from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer
from utils.sound_util import speak
from WinForm.giay_tam_tru import run_gtt

MODEL_PATH = "trained_model"

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng dịch vụ")
        self.root.attributes('-fullscreen', True)

        # Tạo canvas để đặt ảnh nền
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.waiting_window = None
        
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
                print("Đã load encoded_templates thành công")
            else:
                print(f"Không tìm thấy file {encoded_templates_path}")
                # Tạo một encoded_templates trống để tránh lỗi
                self.encoded_templates = {}
        except Exception as e:
            print(f"Lỗi khi load model hoặc encoded_templates: {e}")
            # Fallback: Nếu không load được, tạo model mới
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.encoded_templates = {}
            
        # Load ảnh nền và resize
        bg_image_path = "img/bg_01.jpg"
        if os.path.exists(bg_image_path):
            bg_image = Image.open(bg_image_path)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            bg_image = bg_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)  # Resize ảnh

            # Lưu tham chiếu ảnh vào thuộc tính của lớp
            self.bg_photo = ImageTk.PhotoImage(bg_image)  # Lưu tham chiếu
            self.canvas.create_image(0, 0, image=self.bg_photo, anchor=tk.NW)  # Đảm bảo ảnh được hiển thị
        else:
            print(f"Ảnh không tồn tại: {bg_image_path}")

        self.custom_font = font.Font(family="Helvetica", size=20)

        self.label = tk.Label(self.root, text="CHỌN DỊCH VỤ: ", font=self.custom_font, bg="white", fg="black")
        self.label.place(relx=0.5, rely=0.2, anchor="center")
        
        # Tạo các nút
        self.create_buttons()

        self.mic_button = tk.Button(self.root, text="Bắt đầu nghe", command=self.start_listening,
                                     font=self.custom_font, bg="#4CAF50", fg="white", width=15, height=1)
        self.mic_button.place(relx=0.5, rely=0.8, anchor="center")  # Nút nghe nằm ở dưới cùng của màn hình

        self.exit_button = tk.Button(self.root, text="Thoát", command=self.exit_app,
                                     font=self.custom_font, bg="#f44336", fg="white", width=15, height=1)
        self.exit_button.place(relx=0.5, rely=0.9, anchor="center")  # Nút thoát nằm ở dưới cùng của màn hình

        self.speech = sr.Recognizer()

        # 🔹 Đưa hai nút này sát mép hơn
        self.mic_button.place(relx=0.05, rely=0.92, anchor="w")  # Góc trái sát mép
        self.exit_button.place(relx=0.95, rely=0.92, anchor="e")  # Góc phải sát mép
        
        # Thêm các biến để xử lý xác nhận
        self.confirmed_action = None
        self.confirmation_buttons = []

    def load_actions(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def create_buttons(self):
        actions = [
            "tra cứu bảo hiểm", "cấp lại bằng lái xe", 
            "làm giấy tạm trú", "đăng ký hộ khẩu", 
            "cấp đổi căn cước công dân", "đăng ký kết hôn",
            "khai sinh cho trẻ em", "chứng thực giấy tờ"
        ]

        y_position = 0.35  # Bắt đầu từ vị trí dưới tiêu đề
        x_positions = [0.3, 0.7]  # Cột trái và phải
        button_spacing = 0.15  # Khoảng cách giữa các hàng

        for i, action in enumerate(actions):
            col = x_positions[i % 2]  # Chia đều vào 2 cột (trái, phải)
            button = tk.Button(self.root, text=action, command=lambda a=action: self.perform_action(a),
                            font=self.custom_font, bg="#2196F3", fg="white", width=20, height=2)
            button.place(relx=col, rely=y_position, anchor="center")

            if i % 2 == 1:  # Sau mỗi hàng đủ 2 nút thì xuống dòng
                y_position += button_spacing

    def start_listening(self):
        speak("Tôi đang nghe")
        with sr.Microphone() as source:
            self.label.config(text="Đang nghe...")
            self.root.update()
            audio = self.speech.listen(source)
            try:
                text = self.speech.recognize_google(audio, language='vi-VN')
                self.label.config(text=f"Bạn nói: {text}")
                
                # Thực hiện dự đoán với ngưỡng tin cậy
                best_match, confidence = self.predict_action(text)
                
                # Ngưỡng độ tin cậy để quyết định có hỏi lại hay không
                THRESHOLD = 0.7  # Có thể điều chỉnh ngưỡng này
                
                if best_match and confidence >= THRESHOLD:
                    # Nếu độ tin cậy cao, thực hiện hành động ngay
                    self.perform_action(best_match)
                elif best_match:
                    # Nếu độ tin cậy thấp, hỏi lại người dùng
                    self.ask_for_confirmation(text, best_match)
                else:
                    speak("Xin lỗi, tôi không hiểu yêu cầu của bạn. Vui lòng thử lại.")
                    self.label.config(text="Mời bạn chọn dịch vụ")
                
            except sr.UnknownValueError:
                self.label.config(text="Mời bạn chọn dịch vụ")
            except sr.RequestError:
                self.label.config(text="Lỗi kết nối. Vui lòng thử lại.")

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
        """Hiển thị hộp thoại xác nhận với nút Đồng ý và Huỷ bỏ"""
        # Xóa các nút xác nhận cũ nếu có
        for button in self.confirmation_buttons:
            button.destroy()
        self.confirmation_buttons = []
        
        # Hiển thị thông báo xác nhận
        confirmation_text = f"Bạn muốn {suggested_action} phải không?"
        speak(confirmation_text)
        self.label.config(text=confirmation_text)
        
        # Tạo nút xác nhận
        confirm_button = tk.Button(
            self.root, 
            text="Đúng vậy", 
            command=lambda: self.on_confirmation(suggested_action, True),
            font=self.custom_font, 
            bg="#4CAF50", 
            fg="white", 
            width=15, 
            height=1
        )
        confirm_button.place(relx=0.4, rely=0.3, anchor="center")
        self.confirmation_buttons.append(confirm_button)
        
        # Tạo nút từ chối
        cancel_button = tk.Button(
            self.root, 
            text="Không phải", 
            command=lambda: self.on_confirmation(suggested_action, False),
            font=self.custom_font, 
            bg="#f44336", 
            fg="white", 
            width=15, 
            height=1
        )
        cancel_button.place(relx=0.6, rely=0.3, anchor="center")
        self.confirmation_buttons.append(cancel_button)

    def on_confirmation(self, action, is_confirmed):
        """Xử lý khi người dùng xác nhận hoặc từ chối đề xuất"""
        # Xóa các nút xác nhận
        for button in self.confirmation_buttons:
            button.destroy()
        self.confirmation_buttons = []
        
        if is_confirmed:
            # Nếu người dùng xác nhận, thực hiện hành động
            self.perform_action(action)
        else:
            # Nếu người dùng từ chối, quay lại màn hình chọn
            speak("Vui lòng chọn dịch vụ khác")
            self.label.config(text="CHỌN DỊCH VỤ: ")

    def perform_action(self, command):
        from WinForm.cap_lai_bang_lai_xe import run  # Đảm bảo đường dẫn đúng
        if command == "tra cứu bảo hiểm":
            speak("Bạn muốn tra cứu bảo hiểm gì?")
        elif command == "cấp lại bằng lái xe":
            speak("Mời bạn điền vào form sau!")
            # messagebox.showinfo("Thông báo", "Bản in đang được tạo. Vui lòng chờ...")
            run()
            speak("Bản in đang được tạo. Vui lòng chờ...")
        elif command == "làm giấy tạm trú":
            speak("Bạn điền thông tin vào phiếu khai sau. Sau đó mang đến quầy số 6")
            # messagebox.showinfo("Thông báo", "Bản in đang được tạo. Vui lòng chờ...")
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
            self.label.config(text="Mời bạn chọn dịch vụ")

    def exit_app(self):
        speak("Rất vui được phục vụ bạn, hẹn gặp lại!")
        file_path = "greeting.mp3"
        if os.path.exists(file_path):
            os.remove(file_path)
        self.root.withdraw()  # Ẩn cửa sổ thay vì đóng hoàn toàn để có thể mở lại
        # Quay lại cửa sổ chờ (tạo lại nếu cần)
        self.reset_waiting_window()

    def reset_waiting_window(self):
        if self.waiting_window is None or not self.waiting_window.winfo_exists():
            # Cửa sổ chờ quét thẻ toàn màn hình
            self.waiting_window = tk.Toplevel(self.root)
            self.waiting_window.title("Quét thẻ để tiếp tục")
            self.waiting_window.attributes('-fullscreen', True)  # Fullscreen mode

            screen_width = self.waiting_window.winfo_screenwidth()
            screen_height = self.waiting_window.winfo_screenheight()

            
            self.image = Image.open("img\\bg_00.jpg")
            self.image = self.image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)
            
            self.image_label = tk.Label(self.waiting_window, image=self.photo)
            self.image_label.pack()
            
            self.status_label = tk.Label(self.waiting_window, text="Vui lòng quét thẻ để tiếp tục", font=self.custom_font)
            self.status_label.pack(pady=10)
        else:
            # If waiting window exists, just display it
            self.waiting_window.deiconify()

def run_app(existing_root=None):
    if existing_root:
        # Sử dụng cửa sổ hiện có
        for widget in existing_root.winfo_children():
            widget.destroy()
        existing_root.deiconify()  # Hiển thị lại nếu đã bị ẩn
        app = MainApp(existing_root)
    else:
        # Tạo cửa sổ mới nếu không có cửa sổ nào được truyền vào
        root = tk.Tk()
        app = MainApp(root)
        root.mainloop()

if __name__ == "__main__":
    run_app()