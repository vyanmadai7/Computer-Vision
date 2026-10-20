import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

canvas = np.zeros((480,640,0),dtype=np.uint8)

prev = None
color = (255,0,255)
size = 8

def fingers_up(lm):
    return [
        lm[8].y < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y
    ]

while True:
    _,img = cap.read()
    img = cv2.flip(img,1)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        draw.draw_landmarks(img,hand,mp.solutions.hands.HAND_CONNECTIONS)

        lm = hand.landmark

        h,w,_ = img.shape

        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        index,middle,ring = fingers_up(lm)

        if index and middle and not ring:
            prev = None
            if x < 160:
                color = (255,0,255)
            elif x < 320:
                color = (255,0,0)
            elif x < 480:
                color = (0,255,0)
            else:
                color = (0,255,255)

        elif index and middle and ring:
            if prev:
                cv2.line(canvas,prev,(x,y),(0,0,0),40)
            prev = (x,y)

        elif index and not middle and not ring:
            if prev:
                cv2.line(canvas,prev,(x,y),color,size)
            prev = (x,y)

        else:
            prev = None

    cv2.rectangle(img,(0,0),(160,50),(255,0,255),-1)
    cv2.rectangle(img,(160,0),(320,50),(255,0,0),-1)
    cv2.rectangle(img,(320,0),(480,50),(0,255,0),-1)
    cv2.rectangle(img,(480,0),(640,50),(0,255,255),-1)

    output = cv2.addWeighted(img,0.5,canvas,0.5,0)

    cv2.putText(output,"Draw & Erase", (10,470),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.imshow("Virtual Paint",output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas[:] = 0

    elif key == ord('s'):
        cv2.imwrite("Drawing.png",canvas)
        print("Drawing saved")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
