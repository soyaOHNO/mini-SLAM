import cv2
import numpy as np

def main():
    # --- 設定 ---
    # チェスボードの交点の数（一般的な 9x6 パターン）
    pattern_size = (10, 7) 
    
    # 世界座標系における3Dポイントの準備 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # 3Dポイントと2D画像ポイントを保存する配列
    objpoints = [] # 実世界の3D点
    imgpoints = [] # 画像平面の2D点

    cap = cv2.VideoCapture(0)
    print("【キャリブレーション開始】")
    print("1. チェスボードをカメラに映してください。")
    print("2. 虹色の線が認識されたら、's' キーを押して撮影(保存)します。")
    print("3. カメラの角度や距離を変えながら、15枚ほど撮影してください。")
    print("4. 十分撮れたら 'c' キーを押して計算を開始します。")

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # チェスボードの角を探す
        ret_chess, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        display_img = frame.copy()

        # もし見つかったら描画する
        if ret_chess:
            cv2.drawChessboardCorners(display_img, pattern_size, corners, ret_chess)
            cv2.putText(display_img, "Chessboard Detected! Press 's' to save", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, "Searching for chessboard...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display_img, f"Captured: {count} images", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Calibration', display_img)
        key = cv2.waitKey(1) & 0xFF

        # 's'キーで現在のフレームを保存
        if key == ord('s') and ret_chess:
            # 精度を上げるためにコーナー位置をサブピクセルレベルで精密化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            count += 1
            print(f"[{count}枚目] 撮影完了！")

        # 'c'キーで計算実行
        elif key == ord('c'):
            if count < 10:
                print("※最低でも10枚は撮影してください！")
                continue
            
            print("\n--- 計算中... 少々お待ちください ---")
            h, w = gray.shape
            
            # キャリブレーション実行！
            ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
            print("\n===============================")
            print("【キャリブレーション結果】")
            print("再投影誤差 (小さいほど高精度, 1.0以下が理想):", ret_calib)
            print("\nカメラ行列 (これをSLAMにコピーします):")
            print(repr(mtx)) # Pythonコードとしてコピーしやすい形式で出力
            print("===============================\n")
            break

        # 'q'キーで強制終了
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
