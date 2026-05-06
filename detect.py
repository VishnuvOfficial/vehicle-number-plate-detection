import cv2
import pytesseract
import re
import os
import time
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from collections import deque, Counter
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO(r"C:\Users\vishnu\Documents\vehicle_project\best.pt")

df = pd.read_csv("vehicle_data.csv")

output_file = r"C:\Users\vishnu\Documents\vehicle_project\detected_vehicles.csv"
print(" CSV LOCATION:", output_file)

if not os.path.exists(output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("number,owner,place,time\n")

def clean(text):
    return re.sub(r'[^A-Z0-9]', '', str(text).upper())

def is_valid_plate(text):
    return (
        8 <= len(text) <= 12 and
        sum(c.isdigit() for c in text) >= 4 and
        sum(c.isalpha() for c in text) >= 3
    )

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_match(text):
    best_score = 0
    best_owner = None
    best_place = None

    for _, row in df.iterrows():
        db = clean(row['number'])
        score = similarity(text, db)

        if score > best_score:
            best_score = score
            best_owner = row['owner']
            best_place = row['place']

    if best_score >= 0.75:
        return best_owner, best_place

    return None, None


def save_to_csv(number, owner, place):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{number},{owner},{place},{now}\n")

    print(" SAVED:", number)


cap = cv2.VideoCapture(0)

ocr_buffer = deque(maxlen=5)
seen = {}

print(" ANPR SYSTEM STARTED")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model(frame, conf=0.3)

    h, w = frame.shape[:2]

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pad = 5
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

            plate = frame[y1:y2, x1:x2]
            if plate.size == 0:
                continue

           
            plate = cv2.resize(plate, None, fx=2.5, fy=2.5)
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            text = pytesseract.image_to_string(
                thresh,
                config='--oem 3 --psm 8'
            )

            text = clean(text)
            print("OCR:", text)

            
            if not is_valid_plate(text):
                continue

            
            ocr_buffer.append(text)
            final_text = Counter(ocr_buffer).most_common(1)[0][0]

            owner, place = find_match(final_text)

            if owner is None:
                print(" NO MATCH:", final_text)
                continue

            
            now_time = time.time()

            if final_text not in seen or (now_time - seen[final_text] > 30):
                seen[final_text] = now_time
                save_to_csv(final_text, owner, place)

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, final_text, (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"{owner} | {place}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("ANPR SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()