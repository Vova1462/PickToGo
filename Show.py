import cv2

frame = cv2.imread('/home/vladimir/PycharmProjects/PickToGo/Selection_003.png')
while True:
    cv2.imshow('Picture', frame)
    gframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gframe)
    gx = cv2. Sobel(gframe, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gframe, cv2.CV_32F, 0, 1, ksize=1)
    cv2.imshow('Gradient', gx)
    cv2.imshow('Gradient', gy)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    cv2.imshow('Magnitude', angle)

    if cv2.waitKey() == ord('q'):
        break
