import json
from docx import Document
import os
import subprocess

def run():
        # Đọc dữ liệu từ file JSON
    with open('temp/card_data.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Mở file docx
    doc = Document('Form_bieu_mau\\mau-don-de-nghi-cap-lai-giay-phep-lai-xe.docx')

    # Điền dữ liệu vào file docx mà giữ nguyên định dạng
    for paragraph in doc.paragraphs:
        if '{personName}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{personName}', data['personName'] + "   ")
            for run in paragraph.runs:
                run.font.bold = False  # Không in đậm
                run.font.italic = False  # Không in nghiêng
        if '{nationality}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{nationality}', data['nationality'] + "  ")
            for run in paragraph.runs:
                run.font.bold = False
                run.font.italic = False
        if '{dateOfBirth}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{dateOfBirth}', data['dateOfBirth'] + "  ")
            for run in paragraph.runs:
                run.font.bold = False
                run.font.italic = False
        if '{gender}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{gender}', data['gender'] + "   ")
            for run in paragraph.runs:
                run.font.bold = False
                run.font.italic = False
        if '{residencePlace}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{residencePlace}', data['residencePlace'] + "   ")
            for run in paragraph.runs:
                run.font.bold = False
                run.font.italic = False
        if '{idCode}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{idCode}', data['idCode'] + "   ")
            for run in paragraph.runs:
                run.font.bold = False
                run.font.italic = False
        if '{issueDate}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{issueDate}', data['issueDate'] + "   ")
            for run in paragraph.runs:
                run.font.bold = False
                run.font.italic = False

    # Lưu file docx mới
    doc.save('output.docx')
    # Tên file cần mở
    file_path = 'output.docx'

    # Kiểm tra xem file có tồn tại hay không
    if os.path.exists(file_path):
        # Mở file .docx bằng Microsoft Word
        subprocess.run(["start", "winword", file_path], shell=True)
    else:
        print("File không tồn tại.")
