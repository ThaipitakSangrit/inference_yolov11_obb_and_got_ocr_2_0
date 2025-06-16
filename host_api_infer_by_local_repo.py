import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
from fastapi.responses import JSONResponse
from PIL import Image
import base64
import uvicorn
from ultralytics import YOLO
import cv2
import sys
import os
from transformers import AutoTokenizer
sys.path.append(os.path.abspath('./models/GOT_OCR2_0'))
from models.GOT_OCR2_0.modeling_GOT import GOTQwenForCausalLM

# instantiate fastapi
app = FastAPI()

# Load model YOLOv11
model = YOLO("./models/YOLOv11/best.pt")

# Load GOT OCR model
local_path = "./models/GOT_OCR2_0"

tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
model_got_ocr = GOTQwenForCausalLM.from_pretrained(
    local_path,
    device_map='cuda',
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id,
    trust_remote_code=True
)
model_got_ocr = model_got_ocr.eval().cuda()

# decode base64 function
def decode_base64(image_base64: bytes = None) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(image_base64)))

# pydantic class
class ImageRequest(BaseModel):
    image: bytes = None
    
# ฟังก์ชันช่วย: เรียงลำดับ 4 จุด (ซ้ายบน, ขวาบน, ขวาล่าง, ซ้ายล่าง)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

# prediction api endpoint
@app.post('/predict')
async def predict(request: ImageRequest):  
    try:
        start_time = time.time()  # start timing

        # decode image
        image_rgb = decode_base64(request.image).convert('RGB')
        image_np = np.array(image_rgb)  # ✅ แปลงเป็น NumPy array เพื่อใช้กับ OpenCV
        
        # inference with YOLOv11 model
        results = model(image_rgb)
        
        result = []
        for res in results:
            if res.obb is not None:
                polygons = res.obb.xyxyxyxy.cpu().numpy()  # shape: (N, 8) คือ array ของ 8 จุด (x1, y1, x2, y2, x3, y3, x4, y4)
                # แปลงเป็นกล่องสี่เหลี่ยมตรง (4 จุด)
                boxes = polygons.reshape(-1, 4, 2)  # (N, 4, 2)
                
                # จัดเรียงกล่องสี่เหลี่ยมตามจุดบนสุด
                boxes_sorted = sorted(boxes, key=lambda pts: np.min(pts[:, 1]))
                
                # วนลูปปกติแบบเดิม เพิ่มแค่ order_points เข้าไป
                for i, pts in enumerate(boxes_sorted):
                    # หาจุด 4 จุด ซ้ายบน, ขวาบน, ซ้ายล่าง, ขวาล่าง [มุมองภาพเอียง]
                    pts_src = order_points(np.float32(pts))

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

                    # ✅ แปลงกลับเป็น PIL Image
                    image_cropped = Image.fromarray(warped)
         
                    # inference GOT-OCR
                    content_text = model_got_ocr.chat(tokenizer, image_cropped, ocr_type='ocr-space', gradio_input=True) # gradio_input=True กำหนดเพื่อให้รองรับภาพ PIL Image

                    result.append({
                        "line" : i + 1,
                        "bbox": pts.reshape(-1).astype(int).tolist(),  # ใช้ 8 ค่า (4 จุด)
                        "ocr_result": content_text
                    })

        # calculate processing time
        processing_time = time.time() - start_time

        pred = {
            "result": result,
            "Processing Time (seconds)": processing_time
        }

        return JSONResponse(status_code=200, content=pred)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="localhost", port=2543)