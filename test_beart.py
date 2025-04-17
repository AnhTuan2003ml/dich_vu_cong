import json
import os
import speech_recognition as sr  # type: ignore
import tkinter as tk
from utils.sound_util import speak
import tkinter.font as font
from sentence_transformers import SentenceTransformer
import numpy as np

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Ứng dụng dịch vụ')

        # Set full screen
        self.root.attributes('-fullscreen', True)

        # Customize font
        self.custom_font = font.Font(family="Helvetica", size=16)

        # Create a frame for better organization
        self.frame = tk.Frame(root, bg="#f0f0f0")
        self.frame.pack(expand=True, fill=tk.BOTH)

        # Label
        self.label = tk.Label(self.frame, text="Chọn chức năng bạn muốn:", font=self.custom_font, bg="#f0f0f0")
        self.label.pack(pady=20)

        # Load Sentence-BERT model
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        # Load actions from JSON file
        self.actions = self.load_actions("actions.json")

        # Create buttons dynamically
        for action in self.actions.keys():
            self.create_button(action)

        # Frame for bottom buttons
        self.bottom_frame = tk.Frame(root, bg="#f0f0f0")
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Start Listening button
        self.mic_button = tk.Button(self.bottom_frame, text="Bắt đầu nghe", command=self.start_listening, font=self.custom_font, bg="#4CAF50", fg="white")
        self.mic_button.pack(side=tk.LEFT, padx=20, pady=10)

        # Exit button
        self.exit_button = tk.Button(self.bottom_frame, text="Thoát", command=self.exit_app, font=self.custom_font, bg="#f44336", fg="white")
        self.exit_button.pack(side=tk.RIGHT, padx=20, pady=10)

        self.speech = sr.Recognizer()

        # Mã hóa sẵn các câu mẫu để tăng tốc độ so sánh
        self.encoded_templates = {key: self.model.encode(samples) for key, samples in self.actions.items()}

    def load_actions(self, file_path):
        """
        Đọc danh sách hành động từ file JSON.
        """
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print("Không tìm thấy file actions.json, sử dụng danh sách mặc định.")
            return {}

    def create_button(self, action):
        button = tk.Button(self.frame, text=action, command=lambda: self.perform_action(action), font=self.custom_font, bg="#2196F3", fg="white")
        button.pack(pady=10, padx=20, fill=tk.X)

    def start_listening(self):
        speak("Tôi đang nghe")
        with sr.Microphone() as source:
            self.label.config(text="Đang nghe...")
            self.root.update()  # Update the label immediately
            audio = self.speech.listen(source)

            try:
                text = self.speech.recognize_google(audio, language='vi-VN')
                self.label.config(text=f"Bạn nói: {text}")
                best_match = self.predict_action(text)
                self.perform_action(best_match)
            except sr.UnknownValueError:
                self.label.config(text="Mời bạn chọn dịch vụ")
            except sr.RequestError as e:
                self.label.config(text="Lỗi kết nối. Vui lòng thử lại.")
                print(f"Request error: {e}")

    def predict_action(self, input_text):
        """
        Xác định hành động phù hợp nhất dựa trên độ tương đồng với câu mẫu.
        """
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
        from WinForm.cap_lai_bang_lai_xe import run  # Ensure this is the correct relative path
        if command == "tra cứu bảo hiểm":
            print("Tra cứu bảo hiểm y tế")
            speak("Bạn muốn tra cứu bảo hiểm gì?")

        elif command == "cấp lại bằng lái xe":
            speak("Mời bạn điền vào form sau!")
            run()  # Call the method from the imported module

        elif command == "làm giấy tạm trú":
            print("Làm giấy tạm trú")
            speak("Bạn cần làm giấy tạm trú cho ai?")

    def exit_app(self):
        speak("Rất vui được phục vụ bạn, hẹn gặp lại!")
        self.root.destroy()

def run_main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    run_main()