import cv2
import numpy as np

def feature_extraction(img1, img2):
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return kp1, kp2, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good.append(m)
    return kp1, kp2, good


def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    keyframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = keyframe.shape
    f = 0.8 * w
    cx = w / 2
    cy = h / 2

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    # --- 変更点: OpenCV用の軌跡キャンバス（黒背景）を用意 ---
    traj_img = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # マップの中心下寄りをスタート地点(原点)とする
    start_x, start_y = 300, 500 

    # スタート地点に青い円を描いておく
    cv2.circle(traj_img, (start_x, start_y), 5, (255, 0, 0), -1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp1, kp2, good = feature_extraction(keyframe, img)

        # カメラ映像に特徴点を描画
        display_img = frame.copy()
        for m in good:
            u, v = map(int, kp2[m.trainIdx].pt)
            cv2.circle(display_img, (u, v), 3, (0, 255, 0), -1)
            
        cv2.imshow("Camera View", display_img)

        if len(good) < 20:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # 視差の計算
        parallax = np.mean(np.linalg.norm(pts1 - pts2, axis=1))

        if parallax > 30.0:
            E, mask = cv2.findEssentialMat(pts1, pts2, focal=f, pp=(cx, cy), method=cv2.RANSAC)
            if E is None:
                continue

            pts1 = pts1[mask.ravel() > 0]
            pts2 = pts2[mask.ravel() > 0]

            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=f, pp=(cx, cy))

            # --- 【復活】特徴点の3D復元（三角測量） ---
            P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
            P2 = K @ np.hstack((R, t))
            points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            points3D = points4D[:3] / points4D[3]

            # カメラ前方の点(Z > 0) かつ、遠すぎない点(Z < 50)のみを抽出
            valid = (points3D[2] > 0) & (points3D[2] < 50)
            points3D = points3D[:, valid]

            scale = 50 # 描画スケール（見にくければこの数値を調整してください）

            if points3D.shape[1] > 0:
                # グローバル座標系に変換
                pts_global = R_total @ points3D + t_total
                
                # --- 追加: 2Dマップに特徴点（青色の極小ドット）を打点 ---
                # ::3 などで間引いて描画するとさらに動作が軽くなります
                for i in range(0, pts_global.shape[1], 3): 
                    px = int(pts_global[0, i] * scale) + start_x
                    pz = start_y - int(pts_global[2, i] * scale) # Z軸は前方向なのでY座標から引く
                    
                    # キャンバスの範囲内なら青い点を描画
                    if 0 <= px < 600 and 0 <= pz < 600:
                        cv2.circle(traj_img, (px, pz), 1, (255, 0, 0), -1) 

            # --- 姿勢の更新 ---
            t_total = t_total + R_total @ t
            R_total = R @ R_total
            keyframe = img
            
            # --- 2Dマップにカメラ軌跡（赤色の大きめのドット）を打点 ---
            draw_x = int(t_total[0, 0] * scale) + start_x
            draw_y = start_y - int(t_total[2, 0] * scale) 

            if 0 <= draw_x < 600 and 0 <= draw_y < 600:
                cv2.circle(traj_img, (draw_x, draw_y), 3, (0, 0, 255), -1)
                
            cv2.imshow("Trajectory & Map (Top View)", traj_img)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()