"""
軌跡データの可視化スクリプト
評価実験で記録された軌跡データを確認
作業領域座標系でキーの位置も表示
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os
from pathlib import Path

def load_keyboard_map_work_area(keyboard_map_path='keyboard_map.json', save_json=False):
    """
    keyboard_map.jsonを読み込み、作業領域座標系に変換
    
    Args:
        keyboard_map_path: keyboard_map.jsonのパス
        save_json: 変換結果をJSONファイルに保存するか
    
    Returns:
        dict: 作業領域座標系のキーマップ
    """
    # keyboard_map.jsonを読み込み
    with open(keyboard_map_path, 'r', encoding='utf-8') as f:
        saved_map = json.load(f)
    
    # get_work_area_corners()のロジックを再現
    all_x = []
    all_y = []
    
    for pos in saved_map.values():
        if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
            all_x.append(pos['x'])
            all_y.append(pos['y'])
    
    # 通常キーのサイズを取得（スペースキーを除外）
    key_widths = []
    key_heights = []
    for key, pos in saved_map.items():
        if isinstance(pos, dict) and key != 'space':
            if 'width' in pos:
                key_widths.append(pos['width'])
            if 'height' in pos:
                key_heights.append(pos['height'])
    
    # 通常キーの平均サイズを計算
    avg_key_width = np.mean(key_widths) if key_widths else 0.05
    avg_key_height = np.mean(key_heights) if key_heights else 0.05
    
    # 上下左右1キー分の余白を追加
    margin_x = avg_key_width
    margin_y = avg_key_height
    
    # 4隅の座標を計算（余白付き）
    work_area_corners = np.array([
        [min(all_x) - margin_x, min(all_y) - margin_y],  # 左上
        [max(all_x) + margin_x, min(all_y) - margin_y],  # 右上
        [max(all_x) + margin_x, max(all_y) + margin_y],  # 右下
        [min(all_x) - margin_x, max(all_y) + margin_y]   # 左下
    ], dtype=np.float32)
    
    # ホモグラフィ行列を計算
    dst_corners = np.array([
        [0.0, 0.0],  # 左上
        [1.0, 0.0],  # 右上
        [1.0, 1.0],  # 右下
        [0.0, 1.0]   # 左下
    ], dtype=np.float32)
    
    homography_matrix = cv2.findHomography(
        work_area_corners, 
        dst_corners, 
        cv2.RANSAC,
        ransacReprojThreshold=0.1
    )[0]
    
    # 各キーの座標を変換
    converted_map = {}
    
    for key, pos in saved_map.items():
        # カメラフレーム座標
        camera_x = pos['x']
        camera_y = pos['y']
        
        # ホモグラフィ変換
        src_point = np.array([[[camera_x, camera_y]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, homography_matrix)
        
        work_area_x = dst_point[0][0][0]
        work_area_y = dst_point[0][0][1]
        
        # クリッピング
        work_area_x = np.clip(work_area_x, 0.0, 1.0)
        work_area_y = np.clip(work_area_y, 0.0, 1.0)
        
        # width/heightを作業領域座標系に変換
        key_width_camera = pos.get('width', 0.055)
        key_height_camera = pos.get('height', 0.063)
        
        # 作業領域のサイズ
        work_area_width = np.linalg.norm(work_area_corners[1] - work_area_corners[0])
        work_area_height = np.linalg.norm(work_area_corners[2] - work_area_corners[1])
        
        key_width_wa = key_width_camera / work_area_width
        key_height_wa = key_height_camera / work_area_height
        
        converted_map[key] = {
            'x': float(work_area_x),
            'y': float(work_area_y),
            'width': float(key_width_wa),
            'height': float(key_height_wa)
        }
    
    # オプションでJSONファイルに保存
    if save_json:
        output_file = 'keyboard_map_work_area.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_map, f, indent=2, ensure_ascii=False)
        print(f"[OK] キーマップ（作業領域座標系）を保存しました: {output_file}")
    
    return converted_map

def visualize_trajectory(json_path, keyboard_map_path='keyboard_map.json', show_all_keys=True, save_keyboard_map=False, task_idx=0, input_idx=None, show_all_inputs=False):
    """
    軌跡データを可視化
    
    Args:
        json_path: 評価結果JSONファイルのパス
        keyboard_map_path: キーボードマップファイルのパス
        show_all_keys: 全キーを表示するか
        save_keyboard_map: 変換したキーマップをJSONファイルに保存するか
        task_idx: 表示するタスクのインデックス（0から開始）
        input_idx: 表示する入力のインデックス（0から開始、Noneの場合は全て表示）
        show_all_inputs: 全入力を一度に表示するか（input_idxがNoneの場合に有効）
    """
    
    # キーボードマップ（作業領域座標系）を読み込み
    keyboard_map_wa = load_keyboard_map_work_area(keyboard_map_path, save_json=save_keyboard_map)
    
    # JSONファイル読み込み
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 指定されたタスクと入力
    if task_idx >= len(data['evaluation_log']):
        print(f"[ERROR] タスクインデックス {task_idx} が範囲外です（タスク数: {len(data['evaluation_log'])}）")
        return
    
    task = data['evaluation_log'][task_idx]
    
    # 表示する入力を決定
    if input_idx is None or show_all_inputs:
        input_indices = list(range(len(task['inputs'])))
        print(f"[INFO] 全入力を表示: タスク {task_idx+1}/{len(data['evaluation_log'])}, 入力数: {len(task['inputs'])}")
    else:
        if input_idx >= len(task['inputs']):
            print(f"[ERROR] 入力インデックス {input_idx} が範囲外です（入力数: {len(task['inputs'])}）")
            return
        input_indices = [input_idx]
        print(f"[INFO] 表示: タスク {task_idx+1}/{len(data['evaluation_log'])}, 入力 {input_idx+1}/{len(task['inputs'])}")
    
    # 全入力の軌跡を一度に表示
    if len(input_indices) > 1:
        _visualize_all_trajectories(task, input_indices, keyboard_map_wa, show_all_keys, json_path)
    else:
        _visualize_single_trajectory(task, input_indices[0], keyboard_map_wa, show_all_keys, json_path, task_idx, len(data['evaluation_log']), len(task['inputs']))

def _visualize_all_trajectories(task, input_indices, keyboard_map_wa, show_all_keys, json_path):
    """
    全入力をページめくり形式で表示（矢印キーで移動）
    """
    class TrajectoryViewer:
        def __init__(self, task, input_indices, keyboard_map_wa, show_all_keys, json_path):
            self.task = task
            self.input_indices = input_indices
            self.keyboard_map_wa = keyboard_map_wa
            self.show_all_keys = show_all_keys
            self.json_path = json_path
            self.current_idx = 0
            self.fig = None
            self.ax = None
            
        def draw_trajectory(self):
            """現在の入力の軌跡を描画"""
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(12, 10))
                self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            
            self.ax.clear()
            
            input_idx = self.input_indices[self.current_idx]
            selected_input = self.task['inputs'][input_idx]
            
            # 軌跡データの確認
            if 'trajectory_data' not in selected_input:
                self.ax.text(0.5, 0.5, f"No trajectory data for input {input_idx+1}", 
                            ha='center', va='center', fontsize=16)
                self.fig.canvas.draw()
                return
            
            trajectory = selected_input['trajectory_data']
            xs = [p['finger_x'] for p in trajectory]
            ys = [p['finger_y'] for p in trajectory]
            
            # キーの位置をプロット
            if self.show_all_keys:
                for key, pos in self.keyboard_map_wa.items():
                    key_x = pos['x']
                    key_y = pos['y']
                    key_w = pos['width']
                    key_h = pos['height']
                    
                    rect = plt.Rectangle(
                        (key_x - key_w/2, key_y - key_h/2),
                        key_w, key_h,
                        fill=False, edgecolor='gray', linewidth=1.5, alpha=0.8
                    )
                    self.ax.add_patch(rect)
                    
                    self.ax.text(key_x, key_y, key, ha='center', va='center', 
                               fontsize=9, alpha=0.9, fontweight='bold')
            
            # ターゲットキーと実際に押されたキーの情報
            target_char = selected_input['target_char']
            actual_input = selected_input['actual_input']
            predicted = selected_input['predicted_top3'][0] if selected_input['predicted_top3'] else '?'
            predicted_prob = selected_input['predicted_probs'][0] if selected_input['predicted_probs'] else 0.0
            
            # 正解判定：ターゲットと予測が一致しているか
            is_correct = (predicted.lower() == target_char.lower())
            
            # 正解/不正解で背景色と枠線を変更
            if is_correct:
                bg_color = 'lightgreen'
                border_color = 'green'
                border_width = 4
            else:
                bg_color = 'lightcoral'
                border_color = 'red'
                border_width = 4
            
            # 背景色を設定
            self.ax.set_facecolor(bg_color)
            
            # 枠線を追加
            for spine in self.ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
            
            # 軌跡をプロット（正解/不正解で色を変える）
            trajectory_color = 'darkblue' if is_correct else 'darkred'
            self.ax.plot(xs, ys, color=trajectory_color, linewidth=2, label='Finger Trajectory', alpha=0.7)
            
            # 開始点と終了点をマーク
            self.ax.scatter(xs[0], ys[0], c='green', s=200, 
                          label='Start', zorder=5, marker='o', edgecolors='black', linewidths=2)
            self.ax.scatter(xs[-1], ys[-1], c='red', s=200, 
                          label=f'End (Press: {actual_input})', 
                          zorder=5, marker='X', edgecolors='black', linewidths=2)
            
            # 実際に押したキーの位置をマーク
            if actual_input in self.keyboard_map_wa:
                actual_pos = self.keyboard_map_wa[actual_input]
                self.ax.scatter(actual_pos['x'], actual_pos['y'], c='purple', s=300, 
                              label=f'Actual Key: {actual_input}', 
                              zorder=6, marker='*', edgecolors='yellow', linewidths=2, alpha=0.9)
            
            # ターゲットキーの位置を強調表示
            if target_char in self.keyboard_map_wa:
                target_pos = self.keyboard_map_wa[target_char]
                target_color = 'lime' if is_correct else 'orange'
                target_size = 350 if is_correct else 350
                self.ax.scatter(target_pos['x'], target_pos['y'], c=target_color, s=target_size, 
                              label=f'Target Key: {target_char}', 
                              zorder=7, marker='s', edgecolors='black', linewidths=2, alpha=0.7)
            
            # 予測キーの位置を表示
            if predicted in self.keyboard_map_wa and predicted.lower() != target_char.lower():
                pred_pos = self.keyboard_map_wa[predicted]
                self.ax.scatter(pred_pos['x'], pred_pos['y'], c='cyan', s=250, 
                              label=f'Predicted Key: {predicted}', 
                              zorder=6, marker='D', edgecolors='darkblue', linewidths=2, alpha=0.7)
            
            # タイトル（正解/不正解で色を変える）
            result_mark = '✓ CORRECT' if is_correct else '✗ WRONG'
            title_color = 'darkgreen' if is_correct else 'darkred'
            self.ax.set_title(
                f"Input {input_idx+1}/{len(self.input_indices)} - "
                f"Target: '{target_char}' | Actual: '{actual_input}' | "
                f"Predicted: '{predicted}' ({predicted_prob:.1f}%) - {result_mark}\n"
                f"(←/→: Navigate, Q: Quit)",
                fontsize=14, fontweight='bold', color=title_color
            )
            
            self.ax.set_xlabel('X position (normalized, work area)', fontsize=12)
            self.ax.set_ylabel('Y position (normalized, work area)', fontsize=12)
            self.ax.legend(fontsize=11, loc='upper right')
            self.ax.grid(True, alpha=0.3)
            self.ax.invert_yaxis()
            self.ax.set_xlim(-0.05, 1.05)
            self.ax.set_ylim(1.05, -0.05)
            
            # コンソールに詳細情報を表示
            self._print_input_info(selected_input, input_idx, xs, ys)
            
            self.fig.canvas.draw()
        
        def _print_input_info(self, selected_input, input_idx, xs, ys):
            """入力の詳細情報をコンソールに出力"""
            target_char = selected_input['target_char']
            actual_input = selected_input['actual_input']
            predicted = selected_input['predicted_top3'][0] if selected_input['predicted_top3'] else '?'
            predicted_prob = selected_input['predicted_probs'][0] if selected_input['predicted_probs'] else 0.0
            
            # 正解判定
            is_correct = (predicted.lower() == target_char.lower())
            
            print(f"\n[INFO] Input {input_idx+1}/{len(self.input_indices)}:")
            print(f"   目標文字: '{target_char}'")
            print(f"   実際に押したキー: '{actual_input}'")
            print(f"   予測結果: '{predicted}' ({predicted_prob:.1f}%)")
            print(f"   正解/不正解: {'[OK] 正解' if is_correct else '[NG] 不正解'} (ターゲット='{target_char}' vs 予測='{predicted}')")
            print(f"   軌跡フレーム数: {selected_input['trajectory_length']}")
            print(f"   開始位置: ({xs[0]:.3f}, {ys[0]:.3f})")
            print(f"   終了位置: ({xs[-1]:.3f}, {ys[-1]:.3f})")
            
            # キーまでの距離
            if target_char in self.keyboard_map_wa:
                target_pos = self.keyboard_map_wa[target_char]
                target_dist = np.sqrt((xs[-1] - target_pos['x'])**2 + (ys[-1] - target_pos['y'])**2)
                print(f"   終了点から'{target_char}'（目標）までの距離: {target_dist:.3f}")
            
            if actual_input in self.keyboard_map_wa:
                actual_pos = self.keyboard_map_wa[actual_input]
                actual_dist = np.sqrt((xs[-1] - actual_pos['x'])**2 + (ys[-1] - actual_pos['y'])**2)
                print(f"   終了点から'{actual_input}'（実際に押したキー）までの距離: {actual_dist:.3f}")
        
        def on_key(self, event):
            """キー入力ハンドラー"""
            if event.key == 'right' or event.key == 'down':
                self.current_idx = (self.current_idx + 1) % len(self.input_indices)
                self.draw_trajectory()
            elif event.key == 'left' or event.key == 'up':
                self.current_idx = (self.current_idx - 1) % len(self.input_indices)
                self.draw_trajectory()
            elif event.key == 'q' or event.key == 'escape':
                plt.close(self.fig)
        
        def show(self):
            """表示を開始"""
            self.draw_trajectory()
            print(f"\n[INFO] ページめくり表示")
            print(f"   ←/→ または ↑/↓: 前後の入力に移動")
            print(f"   Q または Esc: 終了")
            plt.show()
    
    viewer = TrajectoryViewer(task, input_indices, keyboard_map_wa, show_all_keys, json_path)
    viewer.show()

def _visualize_single_trajectory(task, input_idx, keyboard_map_wa, show_all_keys, json_path, task_idx, total_tasks, total_inputs):
    """
    単一入力の軌跡を詳細表示
    """
    selected_input = task['inputs'][input_idx]
    
    # 軌跡データを取得
    if 'trajectory_data' not in selected_input:
        print("[ERROR] 軌跡データが保存されていません")
        print("   evaluation_mode.pyを修正後に実験を実施してください")
        return
    
    trajectory = selected_input['trajectory_data']
    xs = [p['finger_x'] for p in trajectory]
    ys = [p['finger_y'] for p in trajectory]
    
    # 可視化
    plt.figure(figsize=(12, 10))
    
    # キーの位置をプロット（デフォルトで全キー表示）
    if show_all_keys:
        # 全キーを表示
        for key, pos in keyboard_map_wa.items():
            key_x = pos['x']
            key_y = pos['y']
            key_w = pos['width']
            key_h = pos['height']
            
            # キーの矩形を描画（濃く表示）
            rect = plt.Rectangle(
                (key_x - key_w/2, key_y - key_h/2),
                key_w, key_h,
                fill=False, edgecolor='gray', linewidth=1.5, alpha=0.8
            )
            plt.gca().add_patch(rect)
            
            # キーのラベル（濃く表示）
            plt.text(key_x, key_y, key, ha='center', va='center', 
                    fontsize=9, alpha=0.9, fontweight='bold')
    
    # ターゲットキーと実際に押されたキーの情報を取得
    target_char = selected_input['target_char']
    actual_input = selected_input['actual_input']
    predicted = selected_input['predicted_top3'][0] if selected_input['predicted_top3'] else '?'
    predicted_prob = selected_input['predicted_probs'][0] if selected_input['predicted_probs'] else 0.0
    
    # 正解判定：ターゲットと予測が一致しているか
    is_correct = (predicted.lower() == target_char.lower())
    
    # 正解/不正解で背景色と枠線を変更
    ax = plt.gca()
    if is_correct:
        bg_color = 'lightgreen'
        border_color = 'green'
        border_width = 4
        trajectory_color = 'darkblue'
    else:
        bg_color = 'lightcoral'
        border_color = 'red'
        border_width = 4
        trajectory_color = 'darkred'
    
    # 背景色を設定
    ax.set_facecolor(bg_color)
    
    # 枠線を追加
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_width)
    
    # 軌跡をプロット
    plt.plot(xs, ys, color=trajectory_color, linewidth=2, label='Finger Trajectory', alpha=0.7)
    
    # 軌跡の進行方向を矢印で表示（10フレームごと）
    for i in range(0, len(xs)-1, 10):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        plt.arrow(xs[i], ys[i], dx*0.5, dy*0.5, 
                 head_width=0.01, head_length=0.01, 
                 fc=trajectory_color, ec=trajectory_color, alpha=0.5)
    
    # 開始点と終了点をマーク
    plt.scatter(xs[0], ys[0], c='green', s=200, 
               label='Start', zorder=5, marker='o', edgecolors='black', linewidths=2)
    plt.scatter(xs[-1], ys[-1], c='red', s=200, 
               label=f'End (Press: {actual_input})', 
               zorder=5, marker='X', edgecolors='black', linewidths=2)
    
    # 最後の5フレームを特別にマーク（認識遅延の確認用）
    last_n = min(5, len(xs))
    if last_n > 1:
        plt.scatter(xs[-last_n:], ys[-last_n:], c='orange', s=100, 
                   label=f'Last {last_n} frames', 
                   zorder=4, marker='s', edgecolors='black', linewidths=1.5, alpha=0.8)
        # 最後の5フレーム間を線で結ぶ
        plt.plot(xs[-last_n:], ys[-last_n:], 'orange', linewidth=2, alpha=0.6, linestyle='--')
    
    # 中間点をプロット
    plt.scatter(xs, ys, c='lightblue', s=20, alpha=0.5, zorder=3)
    
    # 実際に押したキーの位置をマーク（もし記録されていれば）
    if actual_input in keyboard_map_wa:
        actual_pos = keyboard_map_wa[actual_input]
        plt.scatter(actual_pos['x'], actual_pos['y'], c='purple', s=300, 
                   label=f'Actual Key: {actual_input}', 
                   zorder=6, marker='*', edgecolors='yellow', linewidths=2, alpha=0.9)
    
    # ターゲットキーの位置を強調表示
    if target_char in keyboard_map_wa:
        target_pos = keyboard_map_wa[target_char]
        target_color = 'lime' if is_correct else 'orange'
        plt.scatter(target_pos['x'], target_pos['y'], c=target_color, s=350, 
                   label=f'Target Key: {target_char}', 
                   zorder=7, marker='s', edgecolors='black', linewidths=2, alpha=0.7)
    
    # 予測キーの位置を表示（ターゲットと異なる場合）
    if predicted in keyboard_map_wa and predicted.lower() != target_char.lower():
        pred_pos = keyboard_map_wa[predicted]
        plt.scatter(pred_pos['x'], pred_pos['y'], c='cyan', s=250, 
                   label=f'Predicted Key: {predicted}', 
                   zorder=6, marker='D', edgecolors='darkblue', linewidths=2, alpha=0.7)
    
    plt.xlabel('X position (normalized, work area)', fontsize=12)
    plt.ylabel('Y position (normalized, work area)', fontsize=12)
    
    # タイトル（正解/不正解で色を変える）
    result_mark = '✓ CORRECT' if is_correct else '✗ WRONG'
    title_color = 'darkgreen' if is_correct else 'darkred'
    
    plt.title(f"Target: '{target_char}' | Actual Press: '{actual_input}' | Predicted: '{predicted}' ({predicted_prob:.1f}%) - {result_mark}\n"
             f"Trajectory Length: {selected_input['trajectory_length']} frames",
             fontsize=14, fontweight='bold', color=title_color)
    
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Y軸を反転（画面座標系に合わせる）
    
    # 軸の範囲を0-1に固定
    plt.xlim(-0.05, 1.05)
    plt.ylim(1.05, -0.05)
    
    plt.tight_layout()
    
    # 保存
    output_path = json_path.replace('.json', '_trajectory.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 軌跡画像を保存しました: {output_path}")
    
    plt.show()
    
    # 揺れの分析（速度・加速度・方向変化）
    velocities = []
    accelerations = []
    direction_changes = []
    
    for i in range(len(xs) - 1):
        # 速度（フレーム間の距離）
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        vel = np.sqrt(dx**2 + dy**2)
        velocities.append(vel)
        
        # 方向変化（角度）
        if i > 0:
            prev_dx = xs[i] - xs[i-1]
            prev_dy = ys[i] - ys[i-1]
            if prev_dx != 0 or prev_dy != 0:
                prev_angle = np.arctan2(prev_dy, prev_dx)
                curr_angle = np.arctan2(dy, dx)
                angle_diff = abs(curr_angle - prev_angle)
                # 角度差を0-πの範囲に正規化
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                direction_changes.append(angle_diff)
    
    for i in range(len(velocities) - 1):
        # 加速度（速度変化）
        acc = velocities[i+1] - velocities[i]
        accelerations.append(acc)
    
    # 揺れの指標を計算
    avg_velocity = np.mean(velocities) if velocities else 0
    max_velocity = np.max(velocities) if velocities else 0
    avg_acceleration = np.mean(np.abs(accelerations)) if accelerations else 0
    max_acceleration = np.max(np.abs(accelerations)) if accelerations else 0
    avg_direction_change = np.mean(direction_changes) if direction_changes else 0
    max_direction_change = np.max(direction_changes) if direction_changes else 0
    
    # 揺れの振幅（移動方向に対する垂直方向の変動）
    if len(xs) >= 3:
        # 開始点から終了点への直線
        start_to_end_angle = np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])
        
        # 各点から直線までの距離（垂直方向の変動）
        perpendicular_distances = []
        for i in range(len(xs)):
            # 開始点から現在の点へのベクトル
            dx_to_point = xs[i] - xs[0]
            dy_to_point = ys[i] - ys[0]
            
            # 直線方向の単位ベクトル
            line_dir_x = np.cos(start_to_end_angle)
            line_dir_y = np.sin(start_to_end_angle)
            
            # 投影（直線方向成分）
            proj_length = dx_to_point * line_dir_x + dy_to_point * line_dir_y
            
            # 直線方向の位置
            proj_x = xs[0] + proj_length * line_dir_x
            proj_y = ys[0] + proj_length * line_dir_y
            
            # 垂直方向の距離（揺れの振幅）
            perp_dist = np.sqrt((xs[i] - proj_x)**2 + (ys[i] - proj_y)**2)
            perpendicular_distances.append(perp_dist)
        
        max_amplitude = np.max(perpendicular_distances) if perpendicular_distances else 0
        avg_amplitude = np.mean(perpendicular_distances) if perpendicular_distances else 0
    else:
        max_amplitude = 0
        avg_amplitude = 0
    
    # 詳細情報を表示
    print(f"\n[INFO] 詳細情報:")
    print(f"   タスク: {task_idx+1}/{total_tasks}, 入力: {input_idx+1}/{total_inputs}")
    print(f"   目標文字: '{target_char}'")
    print(f"   実際に押したキー: '{actual_input}'")
    print(f"   予測結果: '{predicted}' ({predicted_prob:.1f}%)")
    
    # 正解判定：ターゲットと予測が一致しているか
    is_correct = (predicted.lower() == target_char.lower())
    print(f"   正解/不正解: {'[OK] 正解' if is_correct else '[NG] 不正解'} (ターゲット='{target_char}' vs 予測='{predicted}')")
    print(f"   軌跡フレーム数: {selected_input['trajectory_length']}")
    print(f"   開始位置: ({xs[0]:.3f}, {ys[0]:.3f})")
    print(f"   終了位置（記録された最後のフレーム）: ({xs[-1]:.3f}, {ys[-1]:.3f})")
    print(f"   移動距離: {sum(((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)**0.5 for i in range(len(xs)-1)):.3f}")
    
    # 最後の5フレームの分析（認識遅延の確認）
    last_n = min(5, len(xs))
    if last_n > 1:
        print(f"\n[INFO] 最後の{last_n}フレームの分析（認識遅延確認）:")
        for i in range(last_n):
            frame_idx = len(xs) - last_n + i
            print(f"   フレーム {frame_idx}: ({xs[frame_idx]:.3f}, {ys[frame_idx]:.3f})")
        # 最後の5フレームの平均位置
        last_avg_x = np.mean(xs[-last_n:])
        last_avg_y = np.mean(ys[-last_n:])
        print(f"   最後の{last_n}フレームの平均位置: ({last_avg_x:.3f}, {last_avg_y:.3f})")
        
        # 最後のフレームと平均位置の距離（動きが止まっているか確認）
        movement_in_last = np.sqrt((xs[-1] - xs[-last_n])**2 + (ys[-1] - ys[-last_n])**2)
        print(f"   最後の{last_n}フレーム間の移動距離: {movement_in_last:.3f}")
    
    # キーまでの距離
    if target_char in keyboard_map_wa:
        target_pos = keyboard_map_wa[target_char]
        target_dist = np.sqrt((xs[-1] - target_pos['x'])**2 + (ys[-1] - target_pos['y'])**2)
        print(f"\n   終了点から'{target_char}'（目標）までの距離: {target_dist:.3f}")
    
    if actual_input in keyboard_map_wa:
        actual_pos = keyboard_map_wa[actual_input]
        actual_dist = np.sqrt((xs[-1] - actual_pos['x'])**2 + (ys[-1] - actual_pos['y'])**2)
        print(f"   終了点から'{actual_input}'（実際に押したキー）までの距離: {actual_dist:.3f}")
        
        # もし最後のフレームが実際の押下位置から離れていれば、認識遅延の可能性
        if actual_dist > 0.1:
            print(f"\n   [注意] 終了点が実際の押下キーから離れています（距離: {actual_dist:.3f}）")
            print(f"   → 認識遅延により、押下前の位置が記録されている可能性があります")
    
    # 揺れの分析結果（簡略版）
    is_shaky = (max_direction_change > np.radians(90)) or (avg_amplitude > 0.05) or (max_acceleration > 0.1)
    print(f"\n[INFO] Shake Analysis:")
    print(f"   Shake Detected: {'YES' if is_shaky else 'NO'}")
    print(f"   Max Direction Change: {np.degrees(max_direction_change):.1f} deg")
    print(f"   Max Amplitude: {max_amplitude:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='軌跡データの可視化')
    parser.add_argument('--json', type=str, help='評価結果JSONファイルのパス（指定しない場合は最新のファイルを使用）')
    parser.add_argument('--dir', type=str, default='evaluation_results/P001', 
                       help='評価結果ディレクトリ（デフォルト: evaluation_results/P001）')
    parser.add_argument('--keyboard-map', type=str, default='keyboard_map.json',
                       help='キーボードマップファイルのパス（デフォルト: keyboard_map.json）')
    parser.add_argument('--hide-keys', action='store_true',
                       help='キーの位置を表示しない（デフォルトは表示）')
    parser.add_argument('--save-keyboard-map', action='store_true',
                       help='変換したキーマップをJSONファイルに保存する')
    parser.add_argument('--task', type=int, default=0,
                       help='表示するタスクのインデックス（0から開始、デフォルト: 0）')
    parser.add_argument('--input', type=int, default=None,
                       help='表示する入力のインデックス（0から開始、指定しない場合は全入力表示）')
    parser.add_argument('--all', action='store_true',
                       help='全入力を一度に表示（--inputを無視）')
    
    args = parser.parse_args()
    
    # JSONファイルのパスを決定
    if args.json:
        json_path = args.json
        if not os.path.exists(json_path):
            print(f"[ERROR] ファイルが見つかりません: {json_path}")
            sys.exit(1)
        print(f"[INFO] 指定されたファイルを読み込みます: {json_path}")
    else:
        # 最新のevaluation JSONファイルを探す
        eval_dir = Path(args.dir)
        
        if not eval_dir.exists():
            print(f"[ERROR] {args.dir} ディレクトリが見つかりません")
            sys.exit(1)
        
        # 最新のJSONファイルを取得
        json_files = sorted(eval_dir.glob('evaluation_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not json_files:
            print(f"[ERROR] 評価結果のJSONファイルが見つかりません: {args.dir}")
            sys.exit(1)
        
        json_path = str(json_files[0])
        print(f"[INFO] 最新の評価結果を読み込みます: {json_path}")
    
    visualize_trajectory(json_path, 
                        keyboard_map_path=args.keyboard_map, 
                        show_all_keys=not args.hide_keys,
                        save_keyboard_map=args.save_keyboard_map,
                        task_idx=args.task,
                        input_idx=None if args.all else args.input,
                        show_all_inputs=args.all)

