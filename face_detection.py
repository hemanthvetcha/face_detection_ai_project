import cv2

pathf = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(pathf)

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    faces = face_cascade.detectMultiScale(frame,scaleFactor=1.05, minNeighbors=5)
    for a,b,c,d in faces:
        frame = cv2.rectangle(frame, (a,b), (a+c,b+d), (0,255,0), 3)

    cv2.imshow('Face Detector', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()