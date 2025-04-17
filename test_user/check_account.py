import json

# Đọc dữ liệu người dùng từ file JSON
def load_users_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Đường dẫn tới file JSON chứa thông tin người dùng
file_path = 'test_user\\users.json'  # Thay đổi đường dẫn nếu cần

# Tải dữ liệu người dùng từ file
users_data = load_users_data(file_path)

# Kiểm tra ID có tồn tại trong dữ liệu người dùng
def check_id_exists(cccd):
    return cccd in users_data

# Kiểm tra đăng nhập
def check_login(cccd, mk):
    if users_data[cccd] != mk:
        return {"status_code": 300, "message": "Sai mật khẩu"}
    else:
        return {"status_code": 200, "message": "Đăng nhập thành công"}




