from transformers import AutoModel, AutoTokenizer

local_path = "./models/GOT_OCR2_0"

tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    local_path,
    trust_remote_code=True,
    device_map='cuda',
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id
)
model = model.eval().cuda()

# input your test image
image_file = './test_image/SI53152-A01AGM-04A.jpg'

# plain texts OCR
res = model.chat(tokenizer, image_file, ocr_type='ocr')

# format texts OCR:
# res = model.chat(tokenizer, image_file, ocr_type='format')

# fine-grained OCR:
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# multi-crop OCR:
# res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# render the formatted OCR results:
# res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

print(res)