import cv2
import numpy as np
import time

# ===== YOLO (Step A) =====
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
YOLO_CONF = 0.35
YOLO_IMGSZ = 320
model = YOLO(MODEL_PATH)

def yolo_detect(img_bgr):
    results = model(img_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
    r = results[0]
    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss  = r.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
        dets.append((int(x1), int(y1), int(x2), int(y2), float(cf), int(cl)))
    return dets


# ===== GPIO (Pi 5 safe) using lgpio =====
import lgpio

# Motor pins (BCM)
ENA = 12
IN1 = 5
IN2 = 6

IN3 = 16
IN4 = 19
ENB = 13

PWM_FREQ = 1000  # Hz

# --- Open a working gpiochip (fix for "unknown handle") ---
h = None
for chip in range(0, 8):
    try:
        h = lgpio.gpiochip_open(chip)
        lgpio.gpio_read(h, IN1)  # sanity read
        print(f"[GPIO] Opened /dev/gpiochip{chip}")
        break
    except Exception:
        h = None

if h is None:
    raise RuntimeError("Could not open any /dev/gpiochip*. Run with: sudo -E python lanedrive.py")

# claim outputs
for pin in [ENA, IN1, IN2, IN3, IN4, ENB]:
    lgpio.gpio_claim_output(h, pin, 0)

def pwm_write(pin, duty):
    duty = max(0, min(100, duty))
    lgpio.tx_pwm(h, pin, PWM_FREQ, duty)

def stop():
    lgpio.gpio_write(h, IN1, 0); lgpio.gpio_write(h, IN2, 0)
    lgpio.gpio_write(h, IN3, 0); lgpio.gpio_write(h, IN4, 0)
    pwm_write(ENA, 0)
    pwm_write(ENB, 0)

def drive_right(speed):
    speed = max(-100, min(100, speed))
    if speed >= 0:
        lgpio.gpio_write(h, IN1, 0); lgpio.gpio_write(h, IN2, 1)
        pwm_write(ENA, speed)
    else:
        lgpio.gpio_write(h, IN1, 1); lgpio.gpio_write(h, IN2, 0)
        pwm_write(ENA, -speed)

def drive_left(speed):
    speed = max(-100, min(100, speed))
    if speed >= 0:
        lgpio.gpio_write(h, IN3, 1); lgpio.gpio_write(h, IN4, 0)
        pwm_write(ENB, speed)
    else:
        lgpio.gpio_write(h, IN3, 0); lgpio.gpio_write(h, IN4, 1)
        pwm_write(ENB, -speed)


# ===== Camera =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ===== Control =====
BASE = 25
MIN_BASE = 19
Kp = 0.68
Kd = 0.6   # reduced to stop sudden snap turns

alpha = 0.6
last_error = 0.0
last_raw_error = 0.0

# Anti-snap limits
MAX_TURN = 35
MAX_DERR = 25

# Lane center smoothing
lane_center_f = None
LANE_ALPHA = 0.7

# If your robot turns the WRONG direction, keep this True.
INVERT_STEER = True

lost_count = 0
LOST_SOFT = 12
LOST_HARD = 35

last_left = None
last_right = None
last_width = 160

# ===== STOP behavior tuning =====
STOP_CONFIRM_FRAMES = 2
CLEAR_CONFIRM_FRAMES = 3
stop_seen_count = 0
clear_seen_count = 0
is_stopped_for_object = False

def find_edges_scanline(bin_img, y, midband=18, min_run=1):
    h2, w2 = bin_img.shape
    y = max(0, min(h2-1, y))
    row = bin_img[y, :]

    cx2 = w2 // 2
    left = None
    right = None

    for x in range(cx2-1, 5, -1):
        if x < cx2 - midband and row[x] > 0:
            left = x
            break

    for x in range(cx2+1, w2-5):
        if x > cx2 + midband and row[x] > 0:
            right = x
            break

    return left, right

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[60:240, 0:320]
        h_img, w_img = roi.shape[:2]
        cx = w_img // 2

        # YOLO detection
        detections = yolo_detect(roi)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        scan_y = int(h_img * 0.75)
        left_edge, right_edge = find_edges_scanline(edges, scan_y, midband=22)

        lane_center = None

        if left_edge is not None and right_edge is not None:
            width = right_edge - left_edge
            if 60 < width < 260:
                last_width = width
            lane_center = (left_edge + right_edge) // 2
            lost_count = 0
            last_left, last_right = left_edge, right_edge

        elif left_edge is not None:
            lane_center = left_edge + last_width // 2
            lost_count = 0
            last_left = left_edge

        elif right_edge is not None:
            lane_center = right_edge - last_width // 2
            lost_count = 0
            last_right = right_edge

        else:
            lost_count += 1

            if lost_count <= LOST_SOFT:
                drive_left(25); drive_right(25)
            elif lost_count <= LOST_HARD:
                drive_left(15); drive_right(15)
            else:
                stop()

            cv2.putText(roi, f"LOST {lost_count}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            for x1,y1,x2,y2,cf,cl in detections:
                cv2.rectangle(roi, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(roi, f"cls={cl} {cf:.2f}", (x1, max(15,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            cv2.putText(roi, f"YOLO det={len(detections)}", (5, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

            cv2.imshow("roi", roi)
            cv2.imshow("edges", edges)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ===== Lane center smoothing =====
        if lane_center_f is None:
            lane_center_f = lane_center
        else:
            lane_center_f = int(LANE_ALPHA * lane_center_f + (1 - LANE_ALPHA) * lane_center)
        lane_center = lane_center_f

        # ===== MIO selection (Most Important Object) =====
        mio = None
        if left_edge is not None and right_edge is not None:
            best_h = 0
            for x1,y1,x2,y2,cf,cl in detections:
                obj_cx = (x1 + x2) // 2
                obj_h = (y2 - y1)

                if left_edge < obj_cx < right_edge:
                    if obj_h > best_h:
                        best_h = obj_h
                        mio = (x1,y1,x2,y2,cf,cl)

        # ===== Stop / Resume logic =====
        if mio is not None:
            stop_seen_count += 1
            clear_seen_count = 0
        else:
            clear_seen_count += 1
            stop_seen_count = 0

        if stop_seen_count >= STOP_CONFIRM_FRAMES:
            is_stopped_for_object = True
        if clear_seen_count >= CLEAR_CONFIRM_FRAMES:
            is_stopped_for_object = False

        # ===== Control =====
        raw_error = lane_center - cx
        error = alpha * last_error + (1 - alpha) * raw_error

        d_error = raw_error - last_raw_error
        d_error = max(-MAX_DERR, min(MAX_DERR, d_error))  # clamp derivative kick
        last_raw_error = raw_error
        last_error = error

        turn = (Kp * error) + (Kd * d_error)
        if INVERT_STEER:
            turn = -turn
        turn = max(-MAX_TURN, min(MAX_TURN, turn))        # clamp steering

        slow = min(22, abs(turn) * 0.4)
        base_now = max(MIN_BASE, BASE - slow)

        if is_stopped_for_object:
            base_now = 0

        left_speed  = base_now + turn
        right_speed = base_now - turn

        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))

        drive_left(left_speed)
        drive_right(right_speed)

        # ===== Visual debug =====
        cv2.line(roi, (cx, 0), (cx, h_img), (255, 0, 0), 2)
        cv2.line(roi, (lane_center, 0), (lane_center, h_img), (0, 0, 255), 2)
        cv2.line(roi, (0, scan_y), (w_img, scan_y), (255, 255, 255), 1)

        if left_edge is not None:
            cv2.circle(roi, (left_edge, scan_y), 5, (0,255,0), -1)
        if right_edge is not None:
            cv2.circle(roi, (right_edge, scan_y), 5, (0,255,0), -1)

        for x1,y1,x2,y2,cf,cl in detections:
            cv2.rectangle(roi, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(roi, f"cls={cl} {cf:.2f}", (x1, max(15,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        if mio is not None:
            x1,y1,x2,y2,cf,cl = mio
            cv2.rectangle(roi, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(roi, "MIO", (x1, max(15, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        status = "STOPPED" if is_stopped_for_object else "GO"
        cv2.putText(roi, f"{status}", (240, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if is_stopped_for_object else (0,255,0), 2)

        cv2.putText(roi, f"YOLO det={len(detections)}", (5, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

        cv2.putText(roi, f"err={error:.1f} d={d_error:.1f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
        cv2.putText(roi, f"base={base_now:.0f}  L={left_speed:.0f} R={right_speed:.0f}", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)

        cv2.imshow("roi", roi)
        cv2.imshow("edges", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    try:
        stop()
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()

    try:
        if h is not None:
            lgpio.gpiochip_close(h)
    except Exception:
        pass