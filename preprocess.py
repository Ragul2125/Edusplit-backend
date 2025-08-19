import cv2
from pathlib import Path

# Input folder containing gokul*.jpg
out_dir = Path("output")

# "cleaned" folder inside output
cleaned_dir = out_dir / "cleaned"
cleaned_dir.mkdir(exist_ok=True)

for img_path in out_dir.glob("gokul*.jpg"):
    img = cv2.imread(str(img_path))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15,  # Block size
        9# Constant C
    )

    # Save processed image in "cleaned" folder
    cleaned_path = cleaned_dir / img_path.name
    cv2.imwrite(str(cleaned_path), thresh)

    print(f"Saved cleaned image: {cleaned_path}")
