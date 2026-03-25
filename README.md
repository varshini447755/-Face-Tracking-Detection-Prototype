Face Tracking & Gender Detection Prototype

A real-time computer vision solution that captures video input, detects human faces, identifies gender (Male/Female), and maintains a live headcount. Developed as a technical task for Twite AI Technologies.

🚀 Features
>Real-time Face Detection: Uses Haar Cascade classifiers for low-latency face tracking.
>Gender Identification: Leverages a Pre-trained Caffe Deep Learning model to classify gender.
>Live Headcount: Dynamically tracks and displays the number of people currently in the camera's view.
>Visual Overlay: Draws bounding boxes and text labels directly onto the video stream.

🛠️ Tools Used
>Python 3.12: The core programming language.
>OpenCV (cv2): Primary library for image processing and the DNN (Deep Neural Network) module.
>Haar Cascade: For efficient object detection.
>Caffe Models: Used `gender_deploy.prototxt` (architecture) and `gender_net.caffemodel` (weights) for inference.

💡 Approach
1.Input Stream: Captured frames from the laptop webcam using `cv2.VideoCapture`.
2.Preprocessing: Converted frames to grayscale to optimize the detection speed for the Haar Cascade algorithm.
3.Face Localization: Identified coordinates of faces within the frame.
4.Deep Learning Inference:
   - Cropped the detected face.
   - Converted the crop into a "blob" ($227 \times 227$ pixels) to match the Caffe model's training input.
   - Ran the forward pass through the neural network to get gender probabilities.
5.Output UI: Rendered the bounding boxes, gender labels, and total count onto the live display.

⚠️ Challenges Faced
>Model File Handling: Initial issues with downloading binary `.caffemodel` files from GitHub (resolving corrupted/HTML-pointer files).
>Lighting Sensitivity: Noticed that gender prediction accuracy varies based on ambient lighting and face angles.
>Performance Balancing: Adjusted the `scaleFactor` to ensure smooth frame rates on standard laptop hardware.

📂 Project Structure
├── main.py                 # Main execution script
├── gender_deploy.prototxt  # Model architecture
├── gender_net.caffemodel   # Pre-trained weights
└── README.md               # Project documentation

Please download the .caffemodel file from the Releases section and place it in the project root.