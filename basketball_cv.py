import cv2
import numpy as np

template = cv2.imread('/Users/arushvarun/Documents/SELF-STUDY/DL + CV/WEEK2/OPENCV by ProgrammingKnowledge/SUBMISSION/ball.png',0)
w, h = template.shape[::-1]

cap = cv2.VideoCapture('/Users/arushvarun/Documents/SELF-STUDY/DL + CV/WEEK2/OPENCV by ProgrammingKnowledge/SUBMISSION/volleyball_match.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

output = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(960, 540))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (960, 540))

    #ball
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center = (top_left[0] + w // 2, top_left[1] + h // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    mask = fgbg.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #players
    players = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 10000:
            x, y, w_p, h_p = cv2.boundingRect(cnt)
            players.append((x + w_p / 2, y + h_p / 2))
            cv2.rectangle(frame, (x, y), (x + w_p, y + h_p), (255, 0, 0), 2)

    cv2.putText(frame, f"Players Detected: {len(players)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    output.write(frame)  # inside loop
    cv2.imshow('Ball and Player Tracking', frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

output.release()
cap.release()
cv2.destroyAllWindows()
