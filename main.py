# main.py
import cv2
import mediapipe as mp

from pynput import keyboard

def on_press(key):
    try:
        print(f'キーが押されました: {key.char}')
    except AttributeError:
        print(f'特殊キーが押されました: {key}')

def main():
    # キーボードリスナーの設定
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 既存のカメラ処理
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 手の位置を表示（デバッグ用）
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8:  # 人差し指の先端
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()