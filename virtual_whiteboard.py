import cv2
import numpy as np
import time
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    min_hand_presence_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

wCam, hCam = 640, 480

canvas = np.zeros((hCam, wCam, 3), dtype=np.uint8)

prev_time = 0
px, py = 0, 0
brush_thickness = 5
timestamp = 0

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
]
color_index = 0
color_changed = False

clear_frames = 0
CLEAR_THRESHOLD = 20

SMOOTHING = 0.5
smooth_x, smooth_y = 0, 0

drawing = True

def is_finger_up(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def smooth_position(cx, cy):
    global smooth_x, smooth_y
    if smooth_x == 0 and smooth_y == 0:
        smooth_x, smooth_y = cx, cy
    else:
        smooth_x = smooth_x * SMOOTHING + cx * (1 - SMOOTHING)
        smooth_y = smooth_y * SMOOTHING + cy * (1 - SMOOTHING)
    return int(smooth_x), int(smooth_y)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp += 1
        detection_result = landmarker.detect_for_video(mp_image, timestamp)

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]

            cx, cy = int(landmarks[8].x * wCam), int(landmarks[8].y * hCam)

            index_up = is_finger_up(landmarks, 8, 6)
            middle_up = is_finger_up(landmarks, 12, 10)
            ring_up = is_finger_up(landmarks, 16, 14)
            pinky_up = is_finger_up(landmarks, 20, 18)

            middle_only = middle_up and not index_up and not ring_up and not pinky_up

            fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

            if middle_only:
                cv2.putText(image, "\u064a\u0627 \u0639\u0631\u0635", (wCam // 2 - 100, hCam // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(image, "\u0627\u0642\u0639\u062f \u0639\u0644\u064a \u062c\u0646\u0628", (wCam // 2 - 140, hCam // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                px, py = 0, 0
                smooth_x, smooth_y = 0, 0

            elif fingers_up == 0:
                clear_frames += 1
                progress = clear_frames / CLEAR_THRESHOLD
                cv2.rectangle(image, (10, 10), (int(300 * progress), 40), (0, 0, 255), -1)
                cv2.putText(image, f"CLEARING... {int(progress*100)}%", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if clear_frames >= CLEAR_THRESHOLD:
                    canvas[:] = 0
                    clear_frames = 0
                    cv2.putText(image, "CLEARED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                px, py = 0, 0
                smooth_x, smooth_y = 0, 0
                drawing = True

            else:
                clear_frames = 0

                if index_up and middle_up and ring_up:
                    if not color_changed:
                        color_index = (color_index + 1) % len(colors)
                        color_changed = True
                    cv2.putText(image, "COLOR CHANGED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color_index], 2)
                    px, py = 0, 0
                    smooth_x, smooth_y = 0, 0

                elif index_up and middle_up:
                    color_changed = False
                    drawing = False
                    cv2.putText(image, "PAUSED - MOVE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    px, py = 0, 0
                    smooth_x, smooth_y = 0, 0

                elif index_up:
                    color_changed = False
                    drawing = True
                    sx, sy = smooth_position(cx, cy)
                    
                    if px != 0 and py != 0:
                        cv2.line(canvas, (px, py), (sx, sy), colors[color_index], brush_thickness)
                    cv2.circle(canvas, (sx, sy), brush_thickness // 2, colors[color_index], -1)
                    px, py = sx, sy

                else:
                    color_changed = False
                    px, py = 0, 0
                    smooth_x, smooth_y = 0, 0
                    drawing = True

        else:
            px, py = 0, 0
            color_changed = False
            clear_frames = 0
            smooth_x, smooth_y = 0, 0
            drawing = True

        combined = cv2.addWeighted(image, 1.0, canvas, 0.5, 0)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.rectangle(combined, (10, 45), (150, 75), colors[color_index], -1)
        cv2.putText(combined, f"FPS: {int(fps)}", (15, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not drawing:
            cv2.putText(combined, "PAUSED", (wCam // 2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        y_offset = 10
        instructions = ["Fist: Clear", "Index: Draw", "2 Fingers: Pause", "3 Fingers: Color", "Middle: ???"]
        for text in instructions:
            cv2.rectangle(combined, (wCam - 180, y_offset), (wCam - 10, y_offset + 30), (50, 50, 50), -1)
            cv2.putText(combined, text, (wCam - 170, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 35

        cv2.imshow('Virtual Whiteboard', combined)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
