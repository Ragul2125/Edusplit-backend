from ultralytics import YOLO
from pathlib import Path
import cv2
import os
# import asyncio
model = YOLO("finetunedYolo.pt").to("cuda:0") 

# Define path to the image file
# source = "data/gokul.jpg"

os.makedirs("my_crops", exist_ok=True)

# results = model.predict(source=source)

# semaphore = asyncio.Semaphore(5) 
def splitContent(source:str):
    
    print("Running inference using the **pretrained yolo model**…")
    # async with semaphore:
    try:
        # loop = asyncio.get_event_loop()
        # results = await loop.run_in_executor(None, model.predict, source, False)  # save_crop=False
        results= model.predict(source=source)
        base_name = Path(source).stem
        saved_classes = set()# track class names that have been saved
        save_path=[]
        for i, result in enumerate(results):
            for j, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                crop = result.orig_img[y1:y2, x1:x2]

                class_id = int(result.boxes.cls[j])
                class_name = result.names[class_id]
                class_name_clean = class_name.replace(" ", "_")

                crop_filename = f"{base_name}_{class_name_clean}.jpg"
                crop_path = os.path.join("my_crops", crop_filename)

                if class_name_clean in saved_classes:
                    print(f"Already exists: found duplicate '{class_name_clean}'")
                    return "Duplicate"
                else:
                    cv2.imwrite(crop_path, crop)
                    saved_classes.add(class_name_clean)
                    print(f"Saved: {crop_path}")
                    save_path.append(crop_path)

        print("cropPath:",save_path)
        return save_path
    except Exception:
        return "Error"



# splitContent("data/gokul.jpg")







# # Create your custom output folder
# out_dir = Path("output")
# out_dir.mkdir(exist_ok=True)

# for r in results:
#     im = cv2.imread(r.path)  # original image

#     for j, box in enumerate(r.boxes.xyxy):
#         x1, y1, x2, y2 = map(int, box)

#         cls_id = int(r.boxes.cls[j])
#         cls_name = r.names[cls_id]

#         # Replace spaces with underscores
#         cls_name = cls_name.replace(" ", "_")

#         crop = im[y1:y2, x1:x2]

#         out_path = out_dir / f"gokul_{cls_name}.jpg"

#         counter = 1
#         while out_path.exists():
#             out_path = out_dir / f"gokul_{cls_name}_{counter}.jpg"
#             counter += 1

#         cv2.imwrite(str(out_path), crop)

# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk