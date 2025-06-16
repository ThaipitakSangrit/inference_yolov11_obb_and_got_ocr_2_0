# import library
import os
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

# ---------- ฟังก์ชันตัดภาพตามมุมเอียง ----------
def crop_rotated_box(image_pil, points):
    image_np = np.array(image_pil)
    pts = np.array(points, dtype="float32")

    # จัดเรียง 4 จุด: [top-left, top-right, bottom-right, bottom-left]
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    pts_src = order_points(pts)
    
    # ✅ คำนวณหาความยาวของด้านแต่ละด้าน ของสี่เหลี่ยม
    width_top = np.linalg.norm(pts_src[0] - pts_src[1]) # ความยาว ซ้ายบนถึงขวาบน
    width_bottom = np.linalg.norm(pts_src[2] - pts_src[3]) # ความยาว ซ้ายล่างถึงขวาล่าง
    height_left = np.linalg.norm(pts_src[0] - pts_src[3]) # ความยาว ซ้ายบนถึงซ้ายล่าง
    height_right = np.linalg.norm(pts_src[1] - pts_src[2]) # ความยาว ขวาบนถึงขวาล่าง

    # ✅ หาความกว้างที่สุดและความสูงที่สุด เพื่อแก้ปัญหาขนาดของด้านแต่ลด้านไม่เท่ากัน
    width = max(1, int(max(width_top, width_bottom)))
    height = max(1, int(max(height_left, height_right)))

    # ✅ กำหนดจุดปลายทาง (เป็นแนวตรง)
    pts_dst = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    # ✅ Perspective transform
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image_np, M, (width, height))

    return Image.fromarray(warped)

# ---------- ฟังก์ชันยิง API ----------
def request1(image_path):
    image = Image.open(image_path)
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_base64 = base64.b64encode(buff.getvalue())

    url = f"http://localhost:2543/predict"
    data_send = {
        "image": image_base64.decode("utf-8"),
    }
    response = requests.post(url, json=data_send)
    return response.text

# ---------- วนภาพจากโฟลเดอร์ ----------
image_folder = "./test_image"
result_folder = "./result_image"
os.makedirs(result_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        response_text = request1(image_path)
        print(response_text)

        result = json.loads(response_text)
        image = Image.open(image_path).convert('RGB')

        if 'result' in result:
            for item in result['result']:
                line_number = item['line']
                bbox = item['bbox']
                ocr_result = item['ocr_result']

                # แปลง bbox เป็น array รูป (4, 2)
                pts = np.array(bbox, dtype=np.float32).reshape(4, 2)

                # ตัดภาพตามกรอบเอียงจริง
                cropped = crop_rotated_box(image, pts)

                # ตั้งชื่อไฟล์
                name, _ = os.path.splitext(filename)
                save_path = os.path.join(result_folder, f"{name}_line{line_number}_{ocr_result}.png")
                cropped.save(save_path)
