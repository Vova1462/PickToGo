import cv2
import time
video_capture = cv2.VideoCapture(1)
video_capture.set(3, 1280)
video_capture.set(4, 720)
video_capture.set(10, 0.6)
ret, frame_old = video_capture.read()
i = 0
j = 0
while True:
    time.sleep(0.5)
    ret, frame = video_capture.read()
    diffimg = cv2.absdiff(frame, frame_old) #Просто вычитаем старый и новый кадр
    d_s = cv2.sumElems(diffimg)
    d = (d_s[0]+d_s[1]+d_s[2])/(1280*720)
    frame_old = frame
    print(d)
    if i > 30: #Первые 5-10 кадров камера выходит на режим, их надо пропустить
        if d>1: #Порог срабатывания
            cv2.imwrite("base2/"+str(j)+".jpg", frame)
            j = j+1
    else:
        i = i+1
