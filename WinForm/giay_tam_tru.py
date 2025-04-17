import json
import os
from docx import Document


def run_gtt():
        # Đọc dữ liệu từ file JSON
    with open('temp/card_data.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    os.startfile('WinForm\\Form_bieu_mau\\ct01.docx')
