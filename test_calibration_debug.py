# test_calibration_debug.py
"""
キーボード検出のデバッグ版
エッジ検出の各段階を可視化
"""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.input.calibration.detector import KeyboardDetector
from src.input.calibration.perspective import PerspectiveCorrector


def test_detection_debug():
    """検出処理の各ステップを可視化"""
    
    print("=== キーボード検出デバッグモード ===")
    print("操作方法:")
    print("  1-5: 処理段階の切り替え")
    print("  +/-: Cannyの閾値調整")
    print("  SPACE: 自動検出を試行")
    print("  C: 手動で4隅をクリック")
    print("  ESC: 終了")
    print()
    
    detector = KeyboardDetector()
    corrector = PerspectiveCorrector()
    
    # デバッグ用にパラメータを調整可能に
    detector.canny_low = 30  # より低い閾値から開始
    detector.canny_high = 100
    detector.min_area_ratio = 0.10  # より小さい領域も検出
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        return
    
    # 状態変数
    display_mode = 0  # 0:元画像, 1:グレー, 2:ブラー, 3:エッジ, 4:輪郭
    manual_corners = []
    detected_corners = None
    
    # マウスコールバック
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(manual_corners) < 4:
                manual_corners.append([x, y])
                print(f"Corner {len(manual_corners)}: ({x}, {y})")
                if len(manual_corners) == 4:
                    print("✓ 4隅の指定完了")
    
    cv2.namedWindow('Debug')
    cv2.setMouseCallback('Debug', mouse_callback)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 各処理段階
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, detector.canny_low, detector.canny_high)
            
            # モルフォロジー処理
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 輪郭検出
            contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 表示画像の選択
            if display_mode == 0:
                display = frame.copy()
                mode_text = "Original"
            elif display_mode == 1:
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                mode_text = "Grayscale"
            elif display_mode == 2:
                display = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
                mode_text = "Blurred"
            elif display_mode == 3:
                display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                mode_text = "Edges"
            elif display_mode == 4:
                display = cv2.cvtColor(edges_closed, cv2.COLOR_GRAY2BGR)
                mode_text = "Edges+Morphology"
            else:
                display = frame.copy()
                mode_text = "Contours"
                
                # 輪郭を描画
                if contours:
                    # 面積でソート
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    # 上位5個の輪郭を色分けして表示
                    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255)]
                    
                    h, w = frame.shape[:2]
                    frame_area = h * w
                    
                    for i, cnt in enumerate(contours[:5]):
                        area = cv2.contourArea(cnt)
                        area_ratio = area / frame_area * 100
                        
                        if i < len(colors):
                            cv2.drawContours(display, [cnt], -1, colors[i], 2)
                            
                            # 面積情報を表示
                            x, y, w, h = cv2.boundingRect(cnt)
                            text = f"#{i+1}: {area_ratio:.1f}%"
                            cv2.putText(display, text, (x, y-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            
            # 手動で選択した角を表示
            for i, corner in enumerate(manual_corners):
                cv2.circle(display, tuple(corner), 8, (0, 0, 255), -1)
                cv2.putText(display, str(i+1), (corner[0]+10, corner[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if len(manual_corners) == 4:
                pts = np.array(manual_corners, np.int32)
                cv2.polylines(display, [pts], True, (0, 0, 255), 2)
            
            # 検出された角を表示
            if detected_corners is not None:
                for corner in detected_corners:
                    x, y = int(corner[0]), int(corner[1])
                    cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                pts = np.int32(detected_corners)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            
            # 情報表示
            info_text = [
                f"Mode [{display_mode+1}]: {mode_text}",
                f"Canny: {detector.canny_low}-{detector.canny_high} (+/- to adjust)",
                f"Contours found: {len(contours)}",
                f"Min area: {detector.min_area_ratio*100:.0f}%",
                "Press 1-6 to change mode, SPACE to detect, C to click corners"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display, text, (10, 25 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # 背景を暗く
                cv2.rectangle(display, (8, 10 + i*25), (400, 30 + i*25), (0, 0, 0), -1)
                cv2.putText(display, text, (10, 25 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Debug', display)
            
            # キー処理
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif ord('1') <= key <= ord('6'):
                display_mode = key - ord('1')
                print(f"表示モード: {display_mode + 1}")
            elif key == ord('+'):
                detector.canny_high = min(detector.canny_high + 10, 300)
                detector.canny_low = min(detector.canny_low + 5, detector.canny_high - 10)
                print(f"Canny閾値: {detector.canny_low}-{detector.canny_high}")
            elif key == ord('-'):
                detector.canny_high = max(detector.canny_high - 10, 50)
                detector.canny_low = max(detector.canny_low - 5, 10)
                print(f"Canny閾値: {detector.canny_low}-{detector.canny_high}")
            elif key == ord(' '):  # SPACE: 自動検出
                print("\n自動検出を実行...")
                success, corners = detector.detect_with_perspective(frame)
                if success:
                    detected_corners = corners
                    print("✓ 検出成功")
                    
                    # パースペクティブ補正も試す
                    success, corrected, _, size = corrector.correct_with_auto_size(frame, corners)
                    if success:
                        cv2.imshow('Corrected', corrected)
                        print(f"✓ 補正完了: {size}")
                else:
                    print("✗ 検出失敗")
                    
                    # 最大輪郭の情報を表示
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        area_ratio = cv2.contourArea(largest) / (frame.shape[0] * frame.shape[1])
                        x, y, w, h = cv2.boundingRect(largest)
                        aspect = w / h if h > 0 else 0
                        print(f"  最大輪郭: 面積{area_ratio*100:.1f}%, アスペクト比{aspect:.2f}")
                        print(f"  期待値: 面積15-90%, アスペクト比1.5-4.0")
                        
            elif key == ord('c') or key == ord('C'):  # C: 手動角選択
                manual_corners = []
                print("手動モード: 4隅をクリックしてください")
                
            elif key == ord('r') or key == ord('R'):  # R: リセット
                manual_corners = []
                detected_corners = None
                print("リセットしました")
                
            elif key == ord('p') or key == ord('P'):  # P: 手動角で補正
                if len(manual_corners) == 4:
                    corners_array = np.float32(manual_corners)
                    success, corrected, _, size = corrector.correct_with_auto_size(frame, corners_array)
                    if success:
                        cv2.imshow('Manual Corrected', corrected)
                        print(f"✓ 手動補正完了: {size}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detection_debug()