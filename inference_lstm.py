import threading

import cv2
import keras
import mediapipe as mp
import numpy
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load model
model = keras.models.load_model('model.h5')

lm_list = []

def make_landmark_timestep(results):
    # print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    #Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,connection_drawing_spec= mpDraw.DrawingSpec(color=(21,21,240),thickness=2))

    #vẽ các nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        # print(id,lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx,cy), 5, (0,0,255), cv2.FILLED)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255 ,0)
    thickness = 2
    lineType = 2
    cv2.putText(img,label, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img

def findLabel(results):
    max = - 999
    for val in results[0]:
        if max < val:
            max = val

    for i in range(len(results[0])):
        if results[0][i] == max:
            index = i
    return index
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    index_of_label = findLabel(results)
    if index_of_label == 0:
        label = "BODYSWING"
    if index_of_label == 1:
        label = "LEFTHANDSWING"
    if index_of_label == 2:
        label = "RIGHTHANDSWING"
    return label
label = "...."
i=0
warmup_frames = 60
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results =  pose.process(imgRGB)
    i=i+1

    if i > warmup_frames:
        print("Start detect...")
        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            if len(lm_list) == 10:
                # Đưa vào model nhận diện
                thread_01 = threading.Thread(target=detect, args=(model, lm_list,))
                thread_01.start()
                lm_list = []
                # Vẽ kết quả lên ảnh

            # Vẽ khung xương lên ảnh
            img = draw_landmark_on_image(mpDraw, results, img)
        img = draw_class_on_image(label, img)
        cv2.imshow("image", img)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()