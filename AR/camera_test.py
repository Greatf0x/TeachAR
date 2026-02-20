# camera_test_av.py
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Could not open camera 0 with AVFoundation")
    raise SystemExit
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ok, f = cap.read()
    if not ok:
        print("Frame read failed")
        break
    f = cv2.flip(f, 1)
    cv2.imshow("Mac Camera Test", f)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
