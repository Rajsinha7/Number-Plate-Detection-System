# Number-Plate-Detection-System
The Number Plate Detection System is a real-time computer vision application developed using Python and OpenCV.
The system captures live video from a camera, detects vehicle number plates in each frame, and draws bounding rectangles around the detected plates using a pre-trained Haar Cascade Classifier.
This project can act as a building block for real-world applications such as parking management, traffic monitoring, and law enforcement systems.

--> Features:
 Real-time video capture using camera
 Automatic number plate detection
 Bounding box drawn around detected plates
 Fast and efficient processing
 Uses pre-trained machine learning model
 Works on live video feed

--> Tech Stack
Programming Language: Python
Computer Vision Library: OpenCV
Machine Learning Model: Haar Cascade Classifier (pre-trained)
Camera Interface: OpenCV VideoCapture

--> Project Structure
number-plate-detection/
│
├── haarcascade_russian_plate_number.xml
├── number_plate_detection.py
├── requirements.txt
└── README.md



--> How to Run the Project
1️.Prerequisites

Python 3.x installed

A working webcam

2️. Install Dependencies
pip install opencv-python

3️.Clone the Repository
git clone[ https://github.com/your-username/number-plate-detection.git](https://github.com/Rajsinha7/Number-Plate-Detection-System/tree/main)
cd number-plate-detection

4️. Run the Application
python number_plate_detection.py

5️.Output
Camera window opens
Number plates detected in real time
Rectangles drawn around detected plates
Press q to exit the program

- How It Works
Camera captures live video frames
Frames are converted to grayscale
Haar Cascade Classifier detects number plates

Bounding rectangles are drawn on detected regions

Output is displayed in real time
