from flask import Flask, request, jsonify
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2
import json
import re
import google.generativeai as genai
import tempfile
import os

import torch
print(torch.cuda.is_available())  # should be True
print(torch.version.cuda)   
# ==== CONFIG ====
YOLO_MODEL_PATH = "finetunedYolo.pt"
GEMINI_API_KEY = "AIzaSyBUNiKT6DGuEpIJBBLuj3NNVedb061EEsg"
genai.configure(api_key=GEMINI_API_KEY)

# Load YOLO model once
# yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model =  YOLO(YOLO_MODEL_PATH).to("cuda:0") 
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash")


print(torch.cuda.is_available())  

# This gives the current device name
print(torch.cuda.get_device_name(0))

# This shows the model’s device
print(next(yolo_model.model.parameters()).device)

# ==== PROMPTS ====
PROMPTS = {
    "studentdetailsbox": """You are an OCR and data extraction assistant.
From the given image, identify and return the following fields exactly as written on the paper:
1. Name  
2. Branch  
3. Register No.  
4. Year  
5. Semester  
6. Section  
7. Course Title  
8. Date  
9. Course Code  

Return in JSON:
{
  "Name": "",
  "Branch": "",
  "Register No.": "",
  "Year": "",
  "Semester": "",
  "Section": "",
  "Course Title": "",
  "Date": "",
  "Course Code": ""
}
If empty or not visible, return "".
""",

    "catmarkbox": """You are an OCR and table data extraction assistant.
From the given image of "MARK ENTRY FOR CAT 1 / CAT 2 / CAT 3", return JSON:
{
  "marks": { "1": <val or null or "overwritten">, ... },
  "totals": { "A": <...>, "B": <...>, "C": <...>, "A+B+C": <...> }
}
Rules:
- "-" or tick = null
- empty cell = null
- overwritten numbers = "overwritten"
""",

    "modelmarkbox": """You are an OCR and table data extraction assistant.
From the given image of "MARK ENTRY FOR CAT 4 / CAT 5", return JSON:
{
  "marks": { "1": <val or null or "overwritten">, ... },
  "totals": { "A": <...>, "B": <...>, "C": <...>, "B+C": <...>, "A+B+C": <...> }
}
Same rules as before.
""",

    "totalmarkbox": """You are an OCR assistant.
From the given image showing "Total Marks", return:
{
  "overtotal": "<scored>/<maximum>"
}
Rules:
- Extract exactly what is written.
- Do not guess or calculate.
"""
}

# ==== TYPE DETECTION ====
TYPE_KEYWORDS = {
    "student_details_box": "studentdetailsbox",
    "cat_mark_box": "catmarkbox",
    "model_mark_box": "modelmarkbox",
    "total_mark_box": "totalmarkbox"
}

# ==== HELPERS ====
def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def process_single_image(image_path: str):
    """Process one image with YOLO + Gemini"""
    output_data = {}
    results = yolo_model.predict(source=image_path)

    cleaned_dir = Path("cleaned")
    cleaned_dir.mkdir(exist_ok=True)

    for r in results:
        im = cv2.imread(r.path)
        original_name = Path(r.path).stem

        for j, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(r.boxes.cls[j])
            cls_name = r.names[cls_id].replace(" ", "_")

            # Match YOLO label to prompt type
            img_type = None
            for keyword, type_key in TYPE_KEYWORDS.items():
                if keyword in cls_name.lower():
                    img_type = type_key
                    break
            if img_type is None:
                continue

            # Crop and save
            crop_img = im[y1:y2, x1:x2]
            crop_filename = f"{original_name}_{img_type}_{j+1}.jpg"
            crop_path = cleaned_dir / crop_filename
            cv2.imwrite(str(crop_path), crop_img)

            # Gemini OCR extraction
            prompt = PROMPTS[img_type]
            with Image.open(crop_path) as pil_img:
                response = gemini_model.generate_content([prompt, pil_img])

            clean_text = clean_json_text(response.text)
            try:
                data = json.loads(clean_text)
            except json.JSONDecodeError:
                data = clean_text  # fallback if not valid JSON

            output_data[img_type] = data

    return output_data

# ==== FLASK APP ====
app = Flask(__name__)

@app.route("/process", methods=["POST","GET"])
def process_images():
    if not request.is_json or "images" not in request.json:
        return jsonify({"error": "JSON with 'images' list required"}), 400

    images_list = request.json["images"]
    if not isinstance(images_list, list) or not images_list:
        return jsonify({"error": "'images' should be a non-empty list"}), 400

    final_results = []

    for img_path in images_list:
        if not os.path.exists(img_path):
            final_results.append({"image": img_path, "error": "File not found"})
            continue

        result = process_single_image(img_path)
        final_results.append({"image": img_path, "data": result})

    return jsonify({"results": final_results})

if __name__ == "__main__":
    app.run(debug=True)
