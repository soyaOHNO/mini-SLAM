import cv2
import numpy as np
import open3d as o3d

# 特徴量抽出のセットアップ
orb = cv2.ORB_create(500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    # 最初のキーフレームを初期化
    kf_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kf_kp, kf_des = orb.detectAndCompute(kf_img, None)

    h, w = kf_img.shape

    # カメラ内部パラメータ（キャリブレーション結果をここに直接貼り付ける）
    K = np.array([[639.16832586,   0.        , 250.37877188],
       [  0.        , 636.12418803, 241.89490786],
       [  0.        ,   0.        ,   1.        ]])
    cx, cy = K[0, 2], K[1, 2]
    f = K[0, 0]

    # カメラ姿勢
    R_total = np.eye(3)
    t_total = np.zeros((3,1))

    trajectory = []
    map_points = []
    map_colors = [] # ★追加: 点群の色を保存するリスト

    # --- スケール推定用の辞書 ---
    # 直前のフレームで計算した3D座標を「特徴点のID」で記憶しておく
    p3d_dict = {} 

    # --- Open3Dのリアルタイム描画設定 ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SLAM 3D Viewer - Full Color & Scale Estimation", width=1024, height=768)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    pcd = o3d.geometry.PointCloud()
    lineset = o3d.geometry.LineSet()

    # ダミーデータ（エラー防止）
    pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 1.0]]))
    lineset.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.001]]))
    lineset.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))

    vis.add_geometry(pcd)
    vis.add_geometry(lineset)

    is_view_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_des = orb.detectAndCompute(curr_img, None)

        if kf_des is None or curr_des is None or len(kf_des) < 2 or len(curr_des) < 2:
            continue

        # マッチング
        matches = bf.knnMatch(kf_des, curr_des, k=2)
        good = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        # マッチングの描画
        display_img = frame.copy()
        for m in good:
            u, v = map(int, curr_kp[m.trainIdx].pt)
            cv2.circle(display_img, (u, v), 3, (0, 255, 0), -1)
            
        cv2.imshow("Tracking", display_img)

        if len(good) < 20:
            continue

        pts1 = np.float32([kf_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([curr_kp[m.trainIdx].pt for m in good])

        parallax = np.mean(np.linalg.norm(pts1 - pts2, axis=1))

        if parallax > 30.0:
            E, mask = cv2.findEssentialMat(pts1, pts2, focal=f, pp=(cx, cy), method=cv2.RANSAC)
            if E is None:
                continue

            # 外れ値を除去（RANSACのインライアのみ残す）
            inliers = np.where(mask.ravel() > 0)[0]
            pts1_in = pts1[inliers]
            pts2_in = pts2[inliers]

            _, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, focal=f, pp=(cx, cy))

            # 三角測量
            P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
            P2 = K @ np.hstack((R, t))
            points4D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
            points3D = points4D[:3] / points4D[3]

            # =========================================================
            # ★ 追加①：スケールの動的推定（歩幅の計算）
            # =========================================================
            overlap_prev = []
            overlap_curr = []
            
            # 直前のフレームと今回のフレームで、共通して存在している3D点を探す
            for i, idx in enumerate(inliers):
                q_idx = good[idx].queryIdx # 1つ前のキーフレームでのID
                if q_idx in p3d_dict:
                    overlap_prev.append(p3d_dict[q_idx]) # 前回の3D座標
                    overlap_curr.append(points3D[:, i])  # 今回計算した3D座標

            scale = 1.0 # デフォルトの歩幅
            if len(overlap_prev) >= 2:
                # 共通する点が2つ以上あれば、その点同士の「距離」を比較する
                op_prev = np.array(overlap_prev)
                op_curr = np.array(overlap_curr)
                
                # 点と点の間の距離を計算
                d_prev = np.linalg.norm(op_prev[:-1] - op_prev[1:], axis=1)
                d_curr = np.linalg.norm(op_curr[:-1] - op_curr[1:], axis=1)
                
                # エラー回避
                valid_d = d_curr > 1e-6
                if np.sum(valid_d) > 0:
                    # （前回の距離 / 今回の距離）の比率の中央値をスケール（歩幅）とする
                    scale_ratios = d_prev[valid_d] / d_curr[valid_d]
                    scale = np.median(scale_ratios)
                    # スケールの暴走を防ぐリミッター
                    scale = np.clip(scale, 0.1, 5.0)

            # 計算した歩幅を移動量(t)と3D点群に掛け合わせる
            t = t * scale
            points3D = points3D * scale
            # =========================================================

            # カメラ前方の点 かつ 遠すぎない点（スケールに合わせて上限も変動）
            valid_z = (points3D[2] > 0) & (points3D[2] < 50 * scale)
            valid_indices = np.where(valid_z)[0]

            # 描画を軽くするために間引く（::5）
            sampled_indices = valid_indices[::2] # 2点に1点だけ描画する例（必要に応じて調整してください）

            if len(sampled_indices) > 0:
                pts_to_add = points3D[:, sampled_indices]
                pts_global = R_total @ pts_to_add + t_total
                map_points.append(pts_global.T)

                # =========================================================
                # ★ 追加②：フルカラー3Dマップ化（ピクセルから色を抽出）
                # =========================================================
                colors = []
                for i in sampled_indices:
                    # 画像上の2D座標 (u, v) を取得
                    u, v = int(pts2_in[i, 0]), int(pts2_in[i, 1])
                    # 画像の枠からはみ出さないようにクリップ
                    u = np.clip(u, 0, w-1)
                    v = np.clip(v, 0, h-1)
                    
                    # カメラ画像(frame)から B, G, R の色を取得
                    b, g, r = frame[v, u]
                    
                    # Open3D用に 0.0〜1.0 に変換して保存
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
                    
                map_colors.append(np.array(colors))
                # =========================================================

            # --- 次回のスケール計算のための辞書更新 ---
            new_p3d_dict = {}
            for i in valid_indices:
                t_idx = good[inliers[i]].trainIdx # 今回のフレームでのID
                new_p3d_dict[t_idx] = points3D[:, i]
            p3d_dict = new_p3d_dict

            # 姿勢の更新
            t_total = t_total + R_total @ t
            R_total = R @ R_total

            trajectory.append(t_total.flatten())
            
            # キーフレームの更新
            kf_img = curr_img
            kf_kp = curr_kp
            kf_des = curr_des
            
            # --- Open3D オブジェクトの更新 ---
            if len(map_points) > 0:
                mp_array = np.vstack(map_points)
                mc_array = np.vstack(map_colors) # 色データを結合
                pcd.points = o3d.utility.Vector3dVector(mp_array)
                pcd.colors = o3d.utility.Vector3dVector(mc_array) # 色を適用
                
            if len(trajectory) > 1:
                tr_array = np.array(trajectory)
                lineset.points = o3d.utility.Vector3dVector(tr_array)
                lines = [[i, i+1] for i in range(len(tr_array)-1)]
                lineset.lines = o3d.utility.Vector2iVector(lines)
                lineset.paint_uniform_color([1.0, 0.0, 0.0]) # 軌跡は赤色

            vis.update_geometry(pcd)
            vis.update_geometry(lineset)
            
            if not is_view_initialized:
                vis.reset_view_point(True)
                is_view_initialized = True
                
        vis.poll_events()
        vis.update_renderer()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    vis.destroy_window()

if __name__ == "__main__":
    main()