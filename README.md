# Real-Time Emotion Detection

This project uses a pre-trained model to detect emotions in real-time using webcam feed. The detected emotion is displayed on the video feed.

## Dependencies

- OpenCV
- NumPy
- Keras

You can install the dependencies with the following command:

```bash
pip install opencv-python numpy keras
```

# Usage
Clone this repository.
Download the pre-trained model and place it in the same directory as the script. The model file should be named ‘model.h5’.
Run the script with the command python emotion_detection.py.
The script will start your webcam and begin detecting emotions in real-time. The detected emotion will be displayed on the video feed. Press ‘q’ to quit the program.

# How It Works
The script captures video from the webcam and processes every 10th frame. Each frame is converted to grayscale and faces are detected using Haar cascades. Each detected face is then passed through the pre-trained model to predict the emotion. The emotion label is only updated if it’s different from the previous one, reducing flickering of the emotion label.
