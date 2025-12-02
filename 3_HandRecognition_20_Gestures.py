import cv2
import mediapipe as mp
import math
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from joblib import load
from sklearn.neighbors import KNeighborsClassifier

currentDirectory = f"{Path.cwd()}"
# Load model.
rf:RandomForestClassifier = load(f"{currentDirectory}/RandomForest_4.joblib")
knnModel: KNeighborsClassifier = load(f"{currentDirectory}/KnnModel_4.joblib")

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def scaleLandmarks(landmarks):
    # Take the base of the hand [0] and the center of the hand [5] landmarks as 'base'
    # Where the base is at [0,0] and the center of the hand is at [0, 0.5]
    # Based on that, we transform every other landmark based on that scale.
    scale = abs(landmarks[5][0] - landmarks[0][0]) * 2
    originX = landmarks[0][0]
    originY = landmarks[0][1]

    scaledLandmarks = []
    for i in landmarks:
        scaledX = (i[0] - originX) / scale
        scaledY = (i[1] - originY) / scale
        scaledLandmarks.append([scaledX, scaledY])
    return scaledLandmarks

# Rotates a point based on the origin and how many radians to rotate.
# Returns an array of integers for the new rotated point.
def rotate(origin, point, angle):
    ox = origin[0]
    oy = origin[1]
    px = point[0]
    py = point[1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy] 

# Returns landmarks of the hand normalized with position of fingers between 0 and 1.
def normalizeHand(frame, landmarks):
    imgHeight = frame.shape[0]
    imgWidth = frame.shape[1]

    # Convert the original X, Y coordinates from MediaPipe (Which go between 0 and 1) to the Frame Coordinates on the image (ex: 1920x1080).
    # Ex: if a landmark is at (0.5, 0.5), then in a frame of 1920x1080, the true coordinate is: (960, 540)
    realLandmarks = []
    for i in landmarks:
      x = i[0]
      y = i[1]
      realLandmarks.append([int(imgWidth * x), int(imgHeight * y)])

    # Get angle of MAIN landmarks (Base of hand [0] and center of hand [5].)
    rotatedHand = []
    diffY = realLandmarks[5][1] - realLandmarks[0][1]
    diffX = realLandmarks[5][0] - realLandmarks[0][0]
    angle = math.atan2(diffY, diffX)
    # Rotate the hand so that the MAIN Landmarks are at always at a 90 degree angle and rotate every other landmark respectively.
    for i in range(21):
      rotatedLandmark = rotate(realLandmarks[5], realLandmarks[i], -angle)
      rotatedHand.append(rotatedLandmark) 

    for i in rotatedHand:
      cv2.circle(frame, (int(i[0]), int(i[1])), radius=0, color=(0,0,255), thickness=2)

    # Convert the Frame Coordinates of the hand into normalized values between 0 and 1.
    scaledLandmarks = scaleLandmarks(rotatedHand)
    return scaledLandmarks




def createRow(landmarks):
    data = {}
    for i in range(len(landmarks)):
        data[f"x{i}"] = landmarks[i][0]
        data[f"y{i}"] = landmarks[i][1]
    df = pd.DataFrame(data,index=[0])
    return df

def exportData(data: pd.DataFrame):
    data.to_csv('train.csv', index=False)


waitFrame = 0
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


try:
    labels = ['Paper', 'Rock','V', 'Thumbs Up', 'One','Horn','Scissors','Three Fingers','Pinky','Metal Rock','Paper Inv','Rock Inv','V Inv','One Inv','Horn Inv','Scissors Inv','Three Fingers Inv','Pinky Inv','Metal Rock Inv','Nice.', 'NO HAND']
    rfPredicted = 20
    knnPredicted = 20
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        landmark_list = []
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                landmark_list.append((lm.x, lm.y))
        waitFrame += 1
  
        # Check that ALL landmarks of the hand are detected
        if len(landmark_list) == 21:
            # This 'if' is here to slow things down, for example: 'waitFrame > 60' will execute the code inside the IF Statement ONCE every 60 frames.
            if waitFrame > 20:
                waitFrame = 0
                # The captured Landmarks are normalized. (Rotated and put on a scale of x-y between 0 and 1)
                capturedLandmarks = normalizeHand(frame, landmark_list)
                # Creates a DF Row with the current label and then gets added to the 'dfExportData' Dataframe, where all data is collected with their respective labels.
                newData = createRow(capturedLandmarks)
                rfPredicted = rf.predict(newData)[0]
                knnPredicted = knnModel.predict(newData)[0]
                
        cv2.putText(frame, f"RF: {labels[rfPredicted]}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"KNN: {labels[knnPredicted]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255), thickness=3, lineType=cv2.LINE_AA)
        
        cv2.imshow('Frame', frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()