import cv2
import numpy as np

cap = cv2.VideoCapture("video/video0.MOV")

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

traj = np.zeros((600,600,3), dtype=np.uint8)

R_total = np.eye(3)
t_total = np.zeros((3,1))

ret, prev = cap.read()
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(prev, None)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img, None)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(
        pts1, pts2,
        focal=1.0,
        pp=(0,0),
        method=cv2.RANSAC
    )

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2)

    t_total += R_total @ t
    R_total = R @ R_total

    scale = 100
    x = int(t_total[0][0] * scale) + 300
    z = int(t_total[2][0] * scale) + 100

    cv2.circle(traj, (x,z), 2, (0,255,0), -1)

    cv2.imshow("Trajectory", traj)

    kp1, des1 = kp2, des2

    if cv2.waitKey(1) == 27:
        break