import cv2
import numpy as np
import time
from imutils.object_detection import non_max_suppression
from rplidar import RPLidar
from cv2 import ml



def Calibration(cap):
    i = 0
    board_width, board_height = 6, 9
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    obj_points = []
    img_points = []
    delta_time = 0.5
    last = 0
    flag = True

    while True:
        ret, frame = cap.read()
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, found = cv2.findChessboardCorners(gframe, (board_width, board_height), None)
        now = time.time()
        if i < 19 and now - last >= delta_time and ret:
            obj_points.append(objp)
            corners = cv2.cornerSubPix(gframe, found, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners)
            frame = cv2.drawChessboardCorners(frame, (board_width, board_height), corners, ret)
            i += 1
            last = now

        if i == 19 and flag:
            ret, cam_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,
                                                                            gframe.shape[::-1], None, None)
            height, width = gframe.shape[:2]
            new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, distortion,
                                                                (width, height), 1, (width, height))
            mapx, mapy = cv2.initUndistortRectifyMap(cam_matrix, distortion, None, new_cam_matrix, (width, height), 5)

            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi
            frame = frame[y:y + h, x:x + w]

            mean_error = 0
            for i in range(len(obj_points)):
                img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], cam_matrix, distortion)
                error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                mean_error += error
            print('total_error: ', mean_error / len(obj_points))
            flag = False

        if flag == False:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            frame = frame[y:y + h, x:x + w]

        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            fs = cv2.FileStorage("callibration.xml", cv2.FILE_STORAGE_WRITE)
            if not (fs.isOpened()):
                break
            fs.write('roi', roi)
            fs.write('mapx', mapx)
            fs.write('mapy', mapy)
            fs.release()
            return roi, mapx, mapy
            cv2.destroyAllWindows()
            break


def DetectUsingHaar(frame):
    body = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    lower_body = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

    bodies = body.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=5)
    upper_bodies = upper_body.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=5)
    lower_bodies = lower_body.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

    # print('Found lower_bodies: ', len(lower_bodies))
    # print('Found bodies: ', len(bodies))
    return upper_bodies


def DetectUsingHOGdetector(hog, frame):
    orig = frame.copy()
    rects, weights = hog.detectMultiScale(frame, winStride=(8, 8),
                                          padding=(16, 16), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return orig, frame

# def Caffe_detector(frame):
def RunLidar(lidar):
    iterator = lidar.iter_scans()
    scan = next(iterator)
    offset = np.array([(meas[2]*np.cos(np.radians(meas[1])), meas[2]*np.sin(np.radians(meas[1]))) for meas in scan])

def ComputeDistance(x, offset):
    for x_r, y_r in offset:
        if x_r == x:
            y = y_r



def main():
    flag = False

    lidar = RPLidar('/dev/ttyUSB0')
    info = lidar.get_info()
    health = lidar.get_health()
    print(info, '\n', health)

    cap = cv2.VideoCapture(2)
    if flag:
        roi, mapx, mapy = Calibration(cap)
    else:
        fs = cv2.FileStorage("callibration.xml", cv2.FILE_STORAGE_READ)
        if not (fs.isOpened()):
            return 0
        roi = fs.getNode('roi').mat()
        mapx = fs.getNode('mapx').mat()
        mapy = fs.getNode('mapy').mat()
        fs.release()

    x, y, w, h = roi
    hog = cv2.HOGDescriptor()
    #svm = ml.SVM_create()
    #ml.SVM_load('/HOGlegs70x134.xml')
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hog.load('/home/vladimir/PycharmProjects/PickToGo/HOGlegs.xml')

    #hog.load('/HOGlegs70x134.xml')
    while True:
        #RunLidar(lidar)
        ret, frame = cap.read()
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gframe = cv2.remap(gframe, mapx, mapy, cv2.INTER_LINEAR)
        gframe = gframe[int(y[0]):int(y[0] + h[0]), int(x[0]):int(x[0] + w[0])]


        orig, frame = DetectUsingHOGdetector(hog, gframe)
        # upper_bodies = DetectUsingHaar(gframe)
        # for (x1, y1, w1, h1) in bodies:
        #   cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        # for (x1, y1, w1, h1) in upper_bodies:
        #    cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
        # for (x1, y1, w1, h1) in lower_bodies:
        #        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        cv2.imshow('Detection', frame)
        cv2.imshow('Hog', orig)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
            break


if __name__ == "__main__":
    main()

