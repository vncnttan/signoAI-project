import mediapipe as mp
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import tensorflow as tf
import itertools

recognition_model_path = "./model/kp_classifier.hdf5"
mediapipe_model_path = "D:\BINUS\Assignments\Semester 3\AI\SignoAI\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)

MARGIN = 10  # pixels
HANDEDNESS_FONT_SIZE = 0.3
FONT_SIZE = 0.5
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
PREDICTION_TEXT_COLOR = (88, 54, 205) # red

predictor_model = tf.keras.models.load_model(recognition_model_path)

# Get alphabet mapping
import csv
class_mapping = {}
alphabet_mapping_path = "./model/alphabet_mapping.csv"
with open(alphabet_mapping_path, "r") as mapping_file:
    mapping_reader = csv.reader(mapping_file)
    for row in mapping_reader:
        class_mapping[row[1]] = row[0]

class Mediapipe_Hand_Landmark:
  def __init__(self):
    self.results = None
  
  def draw_landmarks_on_image(self, rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    hand_world_landmarks_list = detection_result.hand_world_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]
      # print(hand_world_landmarks[4])
      
      # Predict using the predictor model.
      hand_world_landmarks = hand_world_landmarks_list[idx]
      print(np.array([preprocess_landmark(hand_world_landmarks)]))
      predict_result = predictor_model.predict(np.array([preprocess_landmark(hand_world_landmarks)]))
      prediction = np.argmax(np.squeeze(predict_result))

      # Draw the hand landmarks.
      hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

      # Get the top left corner of the detected hand's bounding box.
      height, width, _ = annotated_image.shape
      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]
      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - MARGIN

      # Draw handedness (left or right hand) on the image.
      cv2.putText(annotated_image, f"{handedness[0].category_name} hand",
                  (text_x, text_y - 20), cv2.FONT_HERSHEY_DUPLEX,
                  HANDEDNESS_FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
      
      # Draw sign language prediction on the image.
      cv2.putText(annotated_image, f"Sign Language: {class_mapping[str(prediction)]}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, PREDICTION_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

  # Create a hand landmarker instance with the live stream mode:
  def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
      # print('hand landmarker result: {}'.format(result))
      self.results = result
      # print("Output image: {}".format(output_image.numpy_view()))
    
  def main(self): 
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=self.print_result,
        num_hands=2)

    timestamp = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        while video.isOpened(): 
            # Capture frame-by-frame
            ret, frame = video.read()

            if not ret:
                print("Ignoring empty frame")
                break

            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_image, timestamp)

            if(not (self.results is None)):
                mp_image_bgr = self.draw_landmarks_on_image(mp_image.numpy_view(), self.results)
                cv2.imshow('MediaPipe Hand Landmarks', mp_image_bgr)
            else:
                print("Output image is empty")
                
                
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("Closing Camera Stream")
                break

    video.release()
    cv2.destroyAllWindows()


def preprocess_landmark(landmarks):
    temp_landmark_list = [[0, 0, 0] for _ in range(21)]
    for index, landmark in enumerate(landmarks):
        # print(landmark)
        temp_landmark_list[index][0] = landmark.x * 100
        temp_landmark_list[index][1] = landmark.y * 100
        temp_landmark_list[index][2] = landmark.z * 100

    
    # # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

if __name__ == "__main__":
    body_module = Mediapipe_Hand_Landmark()
    body_module.main() 