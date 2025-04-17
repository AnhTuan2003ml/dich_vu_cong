import tkinter as tk
from tkinter import messagebox, PhotoImage
from test_user.check_account import check_login

def create_login_window(id_cccd):
    # Tạo cửa sổ chính
    root = tk.Tk()
    root.title("Đăng nhập")
    
    # Cài đặt kích thước và màu nền
    root.geometry("400x300")
    root.configure(bg="#f0f0f0")

    # Thêm khung cho giao diện
    frame = tk.Frame(root, bg="#ffffff")
    frame.pack(pady=20, padx=20, fill="both", expand=True)

    # Thêm ảnh
    try:
        img = PhotoImage(file="temp\\card_image.jpg")  # Thay thế với đường dẫn đến ảnh của bạn
        img_label = tk.Label(frame, image=img, bg="#ffffff")
        img_label.pack(pady=10)
    except Exception as e:
        print("Lỗi khi tải ảnh:", e)

    # Tạo nhãn và ô nhập cho ID và mật khẩu
    tk.Label(frame, text="ID:", bg="#ffffff", font=("Arial", 12)).pack(pady=5)
    id_entry = tk.Entry(frame, font=("Arial", 12), bd=2, relief="groove")
    id_entry.pack(pady=5)
    id_entry.insert(0, id_cccd)  # Chèn giá trị ID vào ô nhập

    tk.Label(frame, text="Mật khẩu:", bg="#ffffff", font=("Arial", 12)).pack(pady=5)
    password_entry = tk.Entry(frame, show="*", font=("Arial", 12), bd=2, relief="groove")
    password_entry.pack(pady=5)

    login_result = {"status_code": None, "message": None}

    # Hàm kiểm tra đăng nhập
    def login():
        user_id = id_entry.get()
        password = password_entry.get()
        
        res = check_login(user_id, password)
        login_result.update(res)
        if res.get("status_code") == 200:
            root.destroy()
        else:
            messagebox.showerror("Lỗi", res.get("message"))

    # Tạo nút đăng nhập
    login_button = tk.Button(frame, text="Đăng nhập", command=login, bg="#4CAF50", fg="white", font=("Arial", 12))
    login_button.pack(pady=20)

    # Vòng lặp chính để hiển thị cửa sổ
    root.mainloop()
    return login_result
