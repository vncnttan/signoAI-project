# signoAI-project


Sign Language classifier AI

Step by step:
1. Download media pipe hand landmark task model from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
2. Change the model path in the main.py to be your/absolute/path/to/hand_landmarker.task
3. Create sign_detection dataset by running all cells in sign_detection.ipynb
4. Create sign_recognition model by running all cells in sign_recognition.ipynb
5. Run `python main.py`

![Result](image.png)