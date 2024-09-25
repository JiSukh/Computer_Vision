import cv2

cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier('haar_face.xml')

    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('a'):
        break

cv2.destroyAllWindows()
cam.release()