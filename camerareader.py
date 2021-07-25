import cv2 as cv
face_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
video=cv.VideoCapture(0)
while(True):
    check,img=video.read()
    img=cv.flip(img,0)
    g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(g_img, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv.imshow("video",img)
    key=cv.waitKey(1)
    if(key==ord('q')):
        break
video.release()
cv.destroyAllWindows()