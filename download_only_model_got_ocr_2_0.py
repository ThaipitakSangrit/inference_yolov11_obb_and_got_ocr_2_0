from transformers import AutoTokenizer, AutoModel
import os

# โฟลเดอร์ปลายทางที่ต้องการเซฟโมเดล (เปลี่ยน path ได้)
save_dir = "./models/only_model_GOT_OCR2_0"

# โหลด tokenizer และบันทึกลงโฟลเดอร์
tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
tokenizer.save_pretrained(save_dir)

# โหลด model และบันทึกลงโฟลเดอร์
model = AutoModel.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
model.save_pretrained(save_dir)

print(f"Model and tokenizer saved to: {save_dir}")
