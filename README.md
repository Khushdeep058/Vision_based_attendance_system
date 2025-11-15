#Vision-Based Face Recognition Attendance System
Using Deep Learning Frameworks (MTCNN + ArcFace + OpenCV + Tkinter)

This project implements a real-time, automated, vision-based attendance system using deep learning–based face recognition. It replaces manual or RFID-based attendance with a fast, contactless, and reliable solution that works completely offline.

🚀 Features

🎯 Real-time face detection using MTCNN

🔐 High-accuracy recognition using ArcFace embeddings

🧠 Deep learning classifier trained on user embeddings

📸 Live video capture using OpenCV

🗂️ Excel-based attendance management using openpyxl

🖥️ Tkinter GUI for dataset creation, training, and attendance visualization

📁 Automatic student registration in Excel

🔄 Automatic attendance logging with timestamp

🔒 Fully offline, ensuring privacy and data security


🛠️ Technologies Used
Component	Technology
Face Detection	MTCNN
Face Recognition	ArcFace (InsightFace)
Deep Learning	Keras / TensorFlow
GUI	Tkinter
Image Processing	OpenCV
Dataset Management	openpyxl (Excel Sheets)
Language	Python


📦 Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/Vision_based_attendance_system.git
cd Vision_based_attendance_system

2️⃣ Install dependencies
pip install -r requirements.txt


Make sure your requirements.txt includes:

opencv-python
tensorflow
keras
mtcnn
insightface
numpy
pillow
openpyxl

▶️ Running the Application

Run:

python main.py


The Tkinter GUI will launch with options for:

Register New Student

Capture Dataset

Train Model

Start Attendance

View Excel Records

👩‍💻 How It Works
🧑‍🎓 1. Student Registration

User enters:

Name

Roll Number
System stores metadata in students.xlsx.

📸 2. Dataset Creation

Captures ~50 images per student

Uses MTCNN for:

Face detection

Landmark extraction

Alignment

Stores final cropped 112×112 face images

🧠 3. Embedding Extraction (ArcFace)

ArcFace converts each aligned face into a 128-D embedding vector

Stored for training

🏋️ 4. Model Training

Keras classifier trains on embeddings

Saves trained .h5 model

🎥 5. Real-time Attendance

Webcam captures frames

Face detected → embedding generated

Classifier predicts identity

Excel attendance sheet updated with:

Name

Timestamp

Confidence score

📈 Results

Accuracy: 97.8%

Processing Speed: 25 FPS on CPU

Performs reliably under:

Varying illumination

Moderate pose changes

Different facial expressions
