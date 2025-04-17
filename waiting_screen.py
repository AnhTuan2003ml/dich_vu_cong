import os
import json
import tkinter as tk
import tkinter.font as font
import socketio
from PIL import Image, ImageTk
from utils.sound_util import speak
from test_user.check_account import check_id_exists
from WinForm.Gd00 import create_login_window
from main_app import run_app

card_service_socket = socketio.Client()

class WaitingScreen:
    def __init__(self, root):
        self.root = root
        self.root.title('Ứng dụng dịch vụ')
        self.root.attributes('-fullscreen', True)

        self.custom_font = font.Font(family="Helvetica", size=16)

        # Cửa sổ chờ quét thẻ toàn màn hình
        self.waiting_window = tk.Toplevel(self.root)
        self.waiting_window.title("Quét thẻ để tiếp tục")
        self.waiting_window.attributes('-fullscreen', True)  # Fullscreen mode

        screen_width = self.waiting_window.winfo_screenwidth()
        screen_height = self.waiting_window.winfo_screenheight()

        self.image = Image.open("img/bg_00.jpg")
        self.image = self.image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.image)

        self.image_label = tk.Label(self.waiting_window, image=self.photo)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(self.waiting_window, text="Vui lòng quét thẻ để tiếp tục", font=self.custom_font, bg="white")
        self.status_label.place(relx=0.5, rely=0.9, anchor=tk.CENTER)  # Đặt giữa màn hình gần dưới

        self.root.withdraw()  # Ẩn cửa sổ chính đến khi quét thẻ

        card_service_socket.connect("http://192.168.5.1:8000")
        card_service_socket.on("/event", self.handle_card_event)

    def handle_card_event(self, data):
        try:
            event_id = data.get("id")
            if event_id == 2:  # Đọc thẻ thành công
                card_data = data.get("data", {})
                name = card_data.get("personName", "người dùng")
                id_cccd = card_data.get("idCode", "")

                if check_id_exists(id_cccd):
                    speak(f"Xin chào, {name}! Mời bạn đăng nhập để thực hiện dịch vụ!")
                    os.makedirs("temp", exist_ok=True)
                    with open("temp/card_data.json", "w", encoding="utf-8") as f:
                        json.dump(card_data, f, ensure_ascii=False, indent=4)

                    res = create_login_window(id_cccd)  # Hiển thị form đăng nhập
                    if res.get("status_code") == 200:
                        speak(res.get("message"))
                        run_app(self.root)
                else:
                    speak(f"Xin chào, {name}! Bạn chưa có tài khoản, mời bạn đăng kí!")


            elif event_id == 4:  # Lưu ảnh thẻ nếu có
                img_data = data.get("data", {}).get("img_data")
                if img_data:
                    os.makedirs("temp", exist_ok=True)
                    with open("temp/card_image.jpg", "wb") as img_file:
                        import base64
                        img_file.write(base64.b64decode(img_data) if isinstance(img_data, str) else img_data)

        except Exception as e:
            print(f"Lỗi xử lý sự kiện thẻ: {e}")


    def start_main_app(self):
        self.waiting_window.destroy()
        self.root.deiconify()

if __name__ == "__main__":
    root = tk.Tk()
    app = WaitingScreen(root)
    root.mainloop()
