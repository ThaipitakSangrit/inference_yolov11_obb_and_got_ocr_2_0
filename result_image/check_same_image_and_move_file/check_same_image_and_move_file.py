import os
import shutil

full_image_path = r'D:\work\python\ocr_utl3_test_infer_yolov11_obb_and_use_got_ocr_2_0\result_image\check_same_image_and_move_file\original'
cropped_image_path = r'D:\work\python\ocr_utl3_test_infer_yolov11_obb_and_use_got_ocr_2_0\result_image'
output_path = r'D:\work\python\ocr_utl3_test_infer_yolov11_obb_and_use_got_ocr_2_0\result_image\check_same_image_and_move_file\image_same'

cropped_files = os.listdir(cropped_image_path)
cropped_base_names = set()

for file in cropped_files:
    if "_line" in file:
        base = file.split("_line")[0]
        cropped_base_names.add(base)
        
for file in os.listdir(full_image_path):
    base_name, ext = os.path.splitext(file)
    if base_name in cropped_base_names:
        src = os.path.join(full_image_path, file)
        dst = os.path.join(output_path, file)
        shutil.move(src, dst)