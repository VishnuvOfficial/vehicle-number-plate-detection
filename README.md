# 🚗 Vehicle Number Plate Detection System

## 📌 Overview
This project is an AI-based vehicle number plate detection system that identifies vehicles from images or video streams and extracts the number plate text using Optical Character Recognition (OCR).

The system uses a YOLO-based object detection model to detect vehicles and number plates, followed by Tesseract OCR to read the text. The extracted data is then stored in a structured CSV file for further use.

> ⚠️ Note: Owner details are retrieved from a sample dataset for demonstration purposes only.

---

## 🎯 Features
- 🚘 Vehicle detection using YOLO (Ultralytics)
- 🔍 Number plate detection and extraction
- 🔠 Text recognition using Tesseract OCR
- 📊 Data storage in CSV format
- ⏱️ Timestamp-based logging
- 📁 Easy-to-use and modular structure

---

## 🛠️ Tech Stack
- Python  
- OpenCV  
- YOLO (Ultralytics)  
- Tesseract OCR  
- Pandas  

---


---

## ▶️ How to Run

### 1. Install dependencies
pip install opencv-python pytesseract pandas ultralytics


### 2. Install Tesseract OCR
Download and install from:
https://github.com/tesseract-ocr/tesseract

Set path in code:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


### 3. Run the project

python detect.py


---

## 📊 Output
- Extracted number plate text  
- CSV file (`detected_vehicles.csv`) containing detected data  
- Console output with detection details  

---

## 🚀 Future Improvements
- Real-time video stream processing  
- Improved OCR accuracy  
- Web-based dashboard for visualization  
- Deployment using cloud services  

---

## 🤝 Contributing
Feel free to fork this repository and contribute.

---

## 📬 Contact
**Vishnu V**  
Aspiring Data Scientist | AI & ML Enthusiast  


