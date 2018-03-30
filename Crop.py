# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import shutil



drag_start = None
sel = (0,0,0,0)

def onmouse(event, x, y, flags, param):
    global drag_start, sel
    if event == cv.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = 0, 0, 0, 0
    elif event == cv.EVENT_LBUTTONUP:
        if sel[2] > sel[0] and sel[3] > sel[1]:
            patch = gray[sel[1]:sel[3], sel[0]:sel[2]]
            result = cv.matchTemplate(gray,patch,cv.TM_CCOEFF_NORMED)
            cv.imwrite(pos + str(i) + '.jpg', patch)
            cv.imshow("result", patch)
        drag_start = None
    elif drag_start:
        #print flags
        if flags & cv.EVENT_FLAG_LBUTTON:
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
            #img = cv.cvtColor(gray, cv.COLOR_BGR2BGR555)
            img = gray.copy()

            cv.rectangle(img, (sel[0], sel[1]), (sel[2], sel[3]), (0,255,255), 1)
            cv.imshow("gray", img)
        else:
            print("selection is complete")
            drag_start = None

if __name__ == '__main__':
    i = 703
    path ='/home/vladimir/PycharmProjects/PickToGo/final_base/'
    pos = '/home/vladimir/PycharmProjects/PickToGo/pos/'
    neg = '/home/vladimir/PycharmProjects/PickToGo/neg/'
    cv.namedWindow("gray", 1)
    cv.setMouseCallback("gray", onmouse)

    for infile in glob.glob(os.path.join(path, '*.*')):
        ext = os.path.splitext(infile)[1][1:] #get the filename extension
        if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
            print(infile)

            img = cv.imread(infile, 1)
            if img is None:
                continue
            sel = (0, 0, 0, 0)
            drag_start = None
            #gray = cv.cvtColor(img, cv.COLOR_BGR2)
            gray = img.copy()
            cv.imshow("gray", gray)
            if cv.waitKey() == ord('e'):
                i += 1
                continue
            if cv.waitKey() == ord('q'):
                cv.imwrite(neg + str(i) + '.jpg', gray)
                i += 1
                continue
            if cv.waitKey() == 27 or i == 3225:
                break
    cv.destroyAllWindows()