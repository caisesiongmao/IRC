import cv2 as cv
import numpy as np


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = cv.imread('work.jpg')

b, g, r = cv.split(img)
#cv_show('g', g)
v1 = cv.Canny(r, 26, 74)
#v1 = cv.morphologyEx(v1, cv.MORPH_GRADIENT, kernel=3)
subv1 = v1[410:520, 163:250]
subimg = img[410:520, 163:250]
#cv_show('v1', subv1)
contours, hierachy = cv.findContours(subv1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
draw = subimg.copy()
draw = cv.drawContours(draw, contours, 2, (0, 0, 255), 2)
#cv_show('draw', draw)
cnt = contours[2]
x, y, w, h = cv.boundingRect(cnt)
draw_ = img.copy()
res = cv.rectangle(draw_, (x+163, y+410), (x+w+163, y+h+410), (0, 255, 0), 2)
cv.putText(res, f"({str(x+163)},{str(y+410)})", (x+163, y+408),
           cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
cv_show('res', res)
