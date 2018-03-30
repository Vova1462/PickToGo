import cv2
import glob
import os

def main():
    Path = '/home/vladimir/PycharmProjects/PickToGo/neg/'
    save = '/home/vladimir/PycharmProjects/PickToGo/neg1/'
    i = 1
    for infile in glob.glob(os.path.join(Path, '*.*')):
        img = cv2.imread(infile, 1)
        new = cv2.resize(img, (640, 360))
        cv2.imwrite(save+str(i)+'.jpg', new)
        i += 1
main()
