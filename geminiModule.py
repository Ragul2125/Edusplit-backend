import os
import json
from pathlib import Path
import google.generativeai as genai
import re
from PIL import Image
# 1. Configure your API key
genai.configure(api_key="AIzaSyBUNiKT6DGuEpIJBBLuj3NNVedb061EEsg")

# 2. Load image using Pillow
# img_path = "data/abieshwar.jpg"
# img = Image.open(img_path)

# 3. Initialize a Gemini Vision-capable model
model = genai.GenerativeModel(model_name="gemini-2.5-flash")  # or another vision-enabled model

# # 4. Send both prompt and image to the model
# prompt = """You are given OCR text from an exam paper. Extract only the following details for each question:

# question_number – the number of the question (e.g., 1, 2, 3).

# marks_obtained – the marks scored for that question (numeric, can be decimal if applicable).
# If a value is missing or unreadable, put null.

# Additionally, determine if the marks or question number are overwritten or unclear:

# If overwritten/unclear → set "text": "overwritten".

# If clearly written → set "text": "normal".

# Output a JSON array where each object has:


# {
#   "question_number": number | null,
#   "marks_obtained": number | null,
#   "text": "overwritten" | "normal"
# }
# Do not include any other information in the output.
# Be precise and ensure mapping between question_number and marks_obtained is correct.

# If the value is null , put text as normal. if it is overwritten , but it as overwritten"""
# response = model.generate_content([prompt, img])

# # 5. Print the OCR-like extracted text
# print("Extracted Text:\n", response.text)

# Folder paths
input_dir = Path("output")
output_json = Path("json/final_output.json")
output_json.parent.mkdir(exist_ok=True)

# ==== PROMPTS ====
PROMPTS = {
    "studentdetailsbox": """You are given OCR text from a student information section.
Extract the following details:
{
  "name": string | null,
  "register_number": string | null,
  "course_title": string | null,
  "year": string | null,
  "semester": string | null,
  "section": string | null,
  "branch": string | null,
  "date": string | null
}
If a value is missing or unreadable, put null.
Do not include any other keys or text. Return only a valid JSON object.""",

    "catmarkbox": """You are given OCR text from a marks table for a category.
Extract for each question:
{
  "question_number": number | null,
  "marks_obtained": number | null,
  "text": "overwritten" | "normal"
}
If marks or question_number are missing, set text as "normal".
Return only a JSON array with one object per question.""",

    "modelmarkbox": """You are given OCR text from a marks table with multiple questions.
Extract for each question:
{
  "question_number": number | null,
  "marks_obtained": number | null,
  "text": "overwritten" | "normal"
}
If marks or question_number are missing, set text as "normal".
Return only a JSON array with one object per question.""",

    "totalmarkbox": """You are given OCR text showing the total marks.
Extract only:
{
  "total_marks": number / 100 | null
}
Return a valid JSON object with this field only."""
}

# ==== TYPE DETECTION ====
TYPE_KEYWORDS = {
    "student_details_box": "studentdetailsbox",
    "cat_mark_box": "catmarkbox",
    "model_mark_box": "modelmarkbox",
    "total_mark_box": "totalmarkbox"
}

# ==== CLEANING FUNCTION ====
def clean_json_text(text: str) -> str:
    """
    Removes ```json fences and trims whitespace so json.loads can parse it.
    """
    text = text.strip()
    # Remove ```json and ``` markers if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

# ==== MAIN STORAGE ====
# final_data = {}

# # ==== MAIN LOOP ====
# for img_path in input_dir.glob("*.jpg"):
#     img_type = None
#     for keyword, type_key in TYPE_KEYWORDS.items():
#         if keyword in img_path.stem.lower():
#             img_type = type_key
#             break

#     if img_type is None:
#         print(f"⚠ No matching prompt for {img_path.name}, skipping.")
#         continue

#     print(f"🔍 Processing {img_path.name} as {img_type}")

#     img = Image.open(img_path)
#     prompt = PROMPTS[img_type]

#     # Send to Gemini
#     response = model.generate_content([prompt, img])

#     # Clean and parse
#     clean_text = clean_json_text(response.text)
#     try:
#         data = json.loads(clean_text)
#     except json.JSONDecodeError:
#         print(f"⚠ Could not parse JSON for {img_path.name}, saving raw text instead.")
#         data = clean_text

#     # Store in final dictionary
#     final_data[img_type] = data

# # ==== SAVE SINGLE JSON ====
# with open(output_json, "w", encoding="utf-8") as f:
#     json.dump(final_data, f, ensure_ascii=False, indent=2)

# print(f"✅ All results saved in → {output_json}")

def processExamImages(crop_paths):
    """
    crop_paths: str (single file path) OR list of file paths for cropped sections.
    Returns: dict with results grouped for that exam.
    """
    # Ensure input is always a list
    if isinstance(crop_paths, str):
        crop_paths = [crop_paths]

    exam_result = {}
    for img_path in crop_paths:
        img_type = None
        for keyword, type_key in TYPE_KEYWORDS.items():
            if keyword in img_path.lower():
                img_type = type_key
                break

        if img_type is None:
            print(f"⚠ No matching prompt for {img_path}, skipping.")
            continue

        img = Image.open(img_path)
        prompt = PROMPTS[img_type]
        response = model.generate_content([prompt, img])
        clean_text = clean_json_text(response.text)

        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError:
            data = clean_text

        exam_result[img_type] = data

    return exam_result
