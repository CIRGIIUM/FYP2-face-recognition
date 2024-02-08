import cv2 as cv
import os

cap = cv.VideoCapture(2) 

img_prefix = "img" 
id_ = 1 


cv.namedWindow("Feed", cv.WINDOW_NORMAL)
cv.resizeWindow("Feed", 800, 600)

while id_ <= 50:
    _, frame = cap.read()
    cv.imshow("Feed", frame)


    filename = f"{img_prefix}{id_}.jpg"
    cv.imwrite(filename, frame)

    id_ += 1

    if cv.waitKey(1) & 0xFF == ord('x'):
        break


cap.release()
cv.destroyAllWindows()
