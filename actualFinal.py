from yoloModule import splitContent
from geminiModule import processExamImages
from flask import Flask, request, jsonify
import os
import asyncio
import json

from flask_cors import CORS
app=Flask(__name__)


CORS(app) 


JSON_FILE="final_output.json"

@app.route("/upload",methods=['POST'])

def upload():

    os.makedirs("uploads", exist_ok=True)  
    # Expect multipart/form-data
    if "images" not in request.files:
        return jsonify({"error": "No 'images' files uploaded"}), 400
    
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files found in 'images'"}), 400
    

    saved_paths = []
    for file in files:
        save_path = os.path.join("uploads", file.filename)
        file.save(save_path)
        saved_paths.append(save_path)
        print(f"Saved: {save_path}")


    all_crop_paths = {}
    for path in saved_paths:
        crops = splitContent(path)
          # aadesh_student_details_box

            
                
        if "Duplicate" == crops :
            return "The image you uploaded has multiple exam papers"
        
        elif "Error" == crops:
            return "Error while parsing the input image"
        
        for crop in crops:   # loop over the 4 cropped files
            filename = os.path.basename(crop)
            base_name = os.path.splitext(filename)[0]  # aadesh_student_details_box

            all_crop_paths[base_name] = crop

        


    # if "Duplicate" in results:
    #     return "The image you uploaded has multiple exam papers"
    
    # elif "Error" in results:
    #     return "Error while parsing the input image"
    
    final_results = []
    for exam_name, crop_list in all_crop_paths.items():
        print("croplist:",crop_list)
        exam_data = processExamImages(crop_list)
        exam_data["original_image"] = f"{exam_name}.jpg"
        final_results.append(exam_data)

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    return {"status": "success", "files_saved": len(files)}



@app.route("/results", methods=["GET"])
def results():
    if not JSON_FILE:
        return jsonify({"error": "No results found"}), 404
    
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return jsonify(data)



if __name__ == "__main__":
    app.run(debug=True)
