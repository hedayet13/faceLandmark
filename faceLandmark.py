import cv2 as cv
import mediapipe as mp
import time

pTime=0

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh()
drawSpec= mp_drawing.DrawingSpec(thickness=1,circle_radius=2)

cap= cv.VideoCapture(0)
while True:
    ret,frame = cap.read()
    imgRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame,faceLms,mp_face_mesh.FACE_CONNECTIONS,drawSpec,drawSpec)

            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                ih,iw,ic = frame.shape
                x,y = int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv.putText(frame,f'FPS: {int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
    cv.imshow('frame',frame)
    cv.waitKey(1)
