import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
import os

# ============================================================
# CONFIGURATION (No separate config file needed)
# ============================================================
class Config:
    # Use the ultralytics model handle so weights are auto-downloaded at runtime
    MODEL_PATH = "yolov8n-pose"
    CONFIDENCE_THRESHOLD = 0.5
    MAX_TRACKING_AGE = 30
    INPUT_SOURCE = 0
    SAVE_OUTPUT = True
    OUTPUT_PATH = "output_video.avi"
    
    # Keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

# ============================================================
# HUMAN DETECTOR WITH POSE + EYE DETECTION
# ============================================================
class HumanDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        from ultralytics import YOLO
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load model '{model_path}': {e}")
            print("If you're deploying to Vercel, ensure the deployment environment can install PyTorch/Ultralytics or host the weights externally.")
            raise
        self.confidence_threshold = confidence_threshold
        
        # Load eye cascade for eye open/close detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Face cascade for face region detection
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Store previous positions for movement tracking
        self.prev_positions = {}
        self.position_history = {}

    def detect_humans(self, frame):
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            if result.keypoints is None:
                continue

            for idx in range(len(result.boxes)):
                box = result.boxes[idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Get keypoints
                kp = result.keypoints.xy[idx].cpu().numpy()
                kp_conf = result.keypoints.conf[idx].cpu().numpy()

                # Detect eyes open/closed
                eyes_status = self._detect_eyes(gray_frame, kp, kp_conf, (x1, y1, x2, y2))

                # Analyze body pose
                movements = self._analyze_pose(kp, kp_conf, frame.shape[1], frame.shape[0], eyes_status, idx)

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'movements': movements,
                    'keypoints': kp,
                    'keypoints_conf': kp_conf,
                    'eyes_status': eyes_status
                })

        return detections

    def _detect_eyes(self, gray_frame, keypoints, confidences, bbox):
        """Detect if eyes are open or closed"""
        x1, y1, x2, y2 = bbox
        
        # Method 1: Use face region and detect eyes
        face_region = gray_frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return "unknown"
        
        # Get approximate face region from keypoints
        if (confidences[Config.NOSE] > 0.4 and 
            confidences[Config.LEFT_EYE] > 0.4 and 
            confidences[Config.RIGHT_EYE] > 0.4):
            
            nose = keypoints[Config.NOSE]
            left_eye = keypoints[Config.LEFT_EYE]
            right_eye = keypoints[Config.RIGHT_EYE]
            
            # Calculate eye region
            eye_center_y = int((left_eye[1] + right_eye[1]) / 2)
            eye_width = int(abs(left_eye[0] - right_eye[0]))
            
            # Define face ROI around eyes
            face_top = max(0, int(eye_center_y - eye_width))
            face_bottom = min(gray_frame.shape[0], int(nose[1]))
            face_left = max(0, int(right_eye[0] - eye_width * 0.3))
            face_right = min(gray_frame.shape[1], int(left_eye[0] + eye_width * 0.3))
            
            if face_bottom > face_top and face_right > face_left:
                face_roi = gray_frame[face_top:face_bottom, face_left:face_right]
                
                if face_roi.size > 0:
                    # Detect eyes in face region
                    eyes = self.eye_cascade.detectMultiScale(
                        face_roi,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(15, 15)
                    )
                    
                    if len(eyes) >= 1:
                        return "open"
                    else:
                        return "closed"
        
        # Fallback: Use eye keypoint confidence
        left_eye_conf = confidences[Config.LEFT_EYE]
        right_eye_conf = confidences[Config.RIGHT_EYE]
        
        avg_eye_conf = (left_eye_conf + right_eye_conf) / 2
        
        if avg_eye_conf > 0.6:
            return "open"
        elif avg_eye_conf < 0.3:
            return "closed"
        else:
            return "partially_open"

    def _analyze_pose(self, keypoints, confidences, frame_width, frame_height, eyes_status, person_id):
        """Comprehensive pose analysis"""
        movements = []
        
        # Initialize position history for this person
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=15)
        
        # ============================================================
        # 1. CALCULATE BODY MEASUREMENTS
        # ============================================================
        body_height = 0
        torso_height = 0
        leg_height = 0
        body_angle = 0
        
        # Calculate shoulder center
        shoulder_center = None
        if confidences[Config.LEFT_SHOULDER] > 0.4 and confidences[Config.RIGHT_SHOULDER] > 0.4:
            shoulder_center = (keypoints[Config.LEFT_SHOULDER] + keypoints[Config.RIGHT_SHOULDER]) / 2
        
        # Calculate hip center
        hip_center = None
        if confidences[Config.LEFT_HIP] > 0.4 and confidences[Config.RIGHT_HIP] > 0.4:
            hip_center = (keypoints[Config.LEFT_HIP] + keypoints[Config.RIGHT_HIP]) / 2
        
        # Calculate knee center
        knee_center = None
        if confidences[Config.LEFT_KNEE] > 0.4 and confidences[Config.RIGHT_KNEE] > 0.4:
            knee_center = (keypoints[Config.LEFT_KNEE] + keypoints[Config.RIGHT_KNEE]) / 2
        
        # Calculate ankle center
        ankle_center = None
        if confidences[Config.LEFT_ANKLE] > 0.4 and confidences[Config.RIGHT_ANKLE] > 0.4:
            ankle_center = (keypoints[Config.LEFT_ANKLE] + keypoints[Config.RIGHT_ANKLE]) / 2
        
        # Calculate body measurements
        if shoulder_center is not None and hip_center is not None:
            torso_height = np.linalg.norm(shoulder_center - hip_center)
            
            # Calculate body angle from vertical
            body_vector = shoulder_center - hip_center
            vertical = np.array([0, -1])
            
            if np.linalg.norm(body_vector) > 0:
                cos_angle = np.dot(body_vector, vertical) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical))
                body_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        if hip_center is not None and ankle_center is not None:
            leg_height = np.linalg.norm(hip_center - ankle_center)
        
        if confidences[Config.NOSE] > 0.4 and ankle_center is not None:
            body_height = abs(keypoints[Config.NOSE][1] - ankle_center[1])
        elif shoulder_center is not None and hip_center is not None:
            body_height = torso_height * 2.5  # Estimate
        
        # Store current position for movement tracking
        if hip_center is not None:
            self.position_history[person_id].append({
                'hip': hip_center.copy(),
                'time': time.time(),
                'body_height': body_height,
                'body_angle': body_angle
            })
        
        # ============================================================
        # 2. SLEEPING DETECTION (Eyes + Body Position)
        # ============================================================
        is_sleeping = False
        
        # Check if eyes are closed
        if eyes_status == "closed":
            movements.append(('eyes_closed', 0.9))
        elif eyes_status == "open":
            movements.append(('eyes_open', 0.9))
        
        # Check if body is horizontal (lying down)
        if body_angle > 60:  # Body is more horizontal
            is_sleeping = True
            movements.append(('lying_down', 0.95))
            
            if eyes_status == "closed":
                movements.append(('sleeping', 0.98))
            else:
                movements.append(('resting', 0.8))
        
        # ============================================================
        # 3. SITTING vs STANDING DETECTION (Height-Based)
        # ============================================================
        if not is_sleeping:
            if torso_height > 0 and leg_height > 0:
                # Standing: legs are extended, body is vertical
                # Sitting: legs are bent, body height is compressed
                
                height_ratio = leg_height / torso_height if torso_height > 0 else 0
                
                # Check knee angle for sitting
                knee_bent = False
                if (confidences[Config.LEFT_HIP] > 0.4 and 
                    confidences[Config.LEFT_KNEE] > 0.4 and 
                    confidences[Config.LEFT_ANKLE] > 0.4):
                    
                    hip = keypoints[Config.LEFT_HIP]
                    knee = keypoints[Config.LEFT_KNEE]
                    ankle = keypoints[Config.LEFT_ANKLE]
                    
                    # Calculate knee angle
                    knee_angle = self._calculate_angle(hip, knee, ankle)
                    
                    if knee_angle < 130:  # Knee is bent
                        knee_bent = True
                
                # Determine sitting or standing
                if height_ratio < 1.2 or knee_bent:
                    if body_angle < 30:  # Body is still vertical
                        movements.append(('sitting', 0.9))
                    else:
                        movements.append(('crouching', 0.8))
                else:
                    if body_angle < 20:
                        movements.append(('standing', 0.9))
                    else:
                        movements.append(('leaning', 0.7))
            
            elif body_height > 0:
                # Estimate based on body height in frame
                height_ratio = body_height / frame_height
                
                if height_ratio > 0.5:
                    movements.append(('standing', 0.8))
                elif height_ratio > 0.3:
                    movements.append(('sitting', 0.8))
                else:
                    movements.append(('crouching', 0.7))
        
        # ============================================================
        # 4. HAND RAISE DETECTION
        # ============================================================
        # Left hand raise
        if (confidences[Config.LEFT_WRIST] > 0.4 and 
            confidences[Config.LEFT_SHOULDER] > 0.4 and
            confidences[Config.LEFT_ELBOW] > 0.4):
            
            wrist = keypoints[Config.LEFT_WRIST]
            shoulder = keypoints[Config.LEFT_SHOULDER]
            elbow = keypoints[Config.LEFT_ELBOW]
            
            # Check if wrist is above shoulder
            if wrist[1] < shoulder[1] - 30:
                # Check arm angle
                arm_angle = self._calculate_angle(shoulder, elbow, wrist)
                if arm_angle > 120:  # Arm is extended
                    movements.append(('left_hand_raised_high', 0.95))
                else:
                    movements.append(('left_hand_raised', 0.85))
            elif wrist[1] < shoulder[1]:
                movements.append(('left_hand_up', 0.7))
        
        # Right hand raise
        if (confidences[Config.RIGHT_WRIST] > 0.4 and 
            confidences[Config.RIGHT_SHOULDER] > 0.4 and
            confidences[Config.RIGHT_ELBOW] > 0.4):
            
            wrist = keypoints[Config.RIGHT_WRIST]
            shoulder = keypoints[Config.RIGHT_SHOULDER]
            elbow = keypoints[Config.RIGHT_ELBOW]
            
            if wrist[1] < shoulder[1] - 30:
                arm_angle = self._calculate_angle(shoulder, elbow, wrist)
                if arm_angle > 120:
                    movements.append(('right_hand_raised_high', 0.95))
                else:
                    movements.append(('right_hand_raised', 0.85))
            elif wrist[1] < shoulder[1]:
                movements.append(('right_hand_up', 0.7))
        
        # Both hands raised
        if any('left_hand_raised' in m[0] for m in movements) and any('right_hand_raised' in m[0] for m in movements):
            movements.append(('both_hands_raised', 0.95))
        
        # ============================================================
        # 5. WALKING DIRECTION & SPEED DETECTION
        # ============================================================
        if len(self.position_history[person_id]) >= 5 and not is_sleeping:
            history = list(self.position_history[person_id])
            
            # Calculate movement over recent frames
            start_pos = history[0]['hip']
            end_pos = history[-1]['hip']
            time_diff = history[-1]['time'] - history[0]['time']
            
            if time_diff > 0:
                displacement = end_pos - start_pos
                speed = np.linalg.norm(displacement) / time_diff
                
                # Horizontal movement (left/right)
                horizontal_movement = displacement[0]
                vertical_movement = displacement[1]
                
                if abs(horizontal_movement) > 3:
                    if horizontal_movement > 0:
                        movements.append(('moving_right', 0.85))
                    else:
                        movements.append(('moving_left', 0.85))
                
                # Classify movement speed
                if speed > 150:
                    movements.append(('running', 0.9))
                elif speed > 50:
                    movements.append(('walking', 0.85))
                elif speed > 10:
                    movements.append(('slow_walking', 0.7))
                else:
                    if not is_sleeping:
                        movements.append(('stationary', 0.8))
        
        # ============================================================
        # 6. ADDITIONAL POSES
        # ============================================================
        # Waving detection
        if len(self.position_history[person_id]) >= 3:
            # Check for rapid hand movement (waving)
            if (confidences[Config.LEFT_WRIST] > 0.4 or confidences[Config.RIGHT_WRIST] > 0.4):
                # Simple wave detection: hand above shoulder with movement
                if any('hand_raised' in m[0] for m in movements):
                    movements.append(('waving', 0.7))
        
        # Sort by confidence
        movements.sort(key=lambda x: x[1], reverse=True)
        
        return movements
    
    def _calculate_angle(self, a, b, c):
        """Calculate angle at point b between points a and c"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
            return 0
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

# ============================================================
# MOTION TRACKER
# ============================================================
class MotionTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.bboxes = {}
        self.detections = {}
        self.motion_status = {}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, detection):
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.bboxes[object_id] = bbox
        self.detections[object_id] = detection
        self.disappeared[object_id] = 0
        self.motion_status[object_id] = "Unknown"
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]
        del self.detections[object_id]
        del self.motion_status[object_id]

    def update(self, tracking_input):
        if len(tracking_input) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([t[1] for t in tracking_input])
        input_bboxes = [t[0] for t in tracking_input]
        input_detections = [t[2] for t in tracking_input]

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_detections[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.detections[object_id] = input_detections[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col], input_detections[col])

        return self.objects

    def get_motion_info(self, object_id):
        if object_id not in self.objects:
            return None
        return {
            'id': object_id,
            'position': self.objects[object_id],
            'bbox': self.bboxes[object_id],
            'detection': self.detections[object_id],
            'status': self.motion_status[object_id]
        }

# ============================================================
# MAIN DETECTION SYSTEM
# ============================================================
class HumanMotionDetectionSystem:
    def __init__(self):
        self.detector = HumanDetector(
            Config.MODEL_PATH,
            Config.CONFIDENCE_THRESHOLD
        )
        self.tracker = MotionTracker(
            max_disappeared=Config.MAX_TRACKING_AGE,
            max_distance=100
        )

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Movement colors
        self.movement_colors = {
            'sleeping': (128, 0, 128),       # Purple
            'lying_down': (128, 0, 128),     # Purple
            'resting': (180, 100, 180),      # Light purple
            'sitting': (0, 165, 255),        # Orange
            'standing': (255, 255, 255),     # White
            'crouching': (100, 100, 100),    # Gray
            'walking': (0, 255, 0),          # Green
            'slow_walking': (100, 255, 100), # Light green
            'running': (0, 255, 255),        # Yellow
            'moving_left': (255, 100, 100),  # Light blue
            'moving_right': (100, 100, 255), # Light red
            'left_hand_raised': (0, 0, 255), # Red
            'right_hand_raised': (0, 0, 255),# Red
            'left_hand_raised_high': (0, 0, 255),
            'right_hand_raised_high': (0, 0, 255),
            'both_hands_raised': (0, 0, 255),# Red
            'waving': (255, 0, 255),         # Magenta
            'eyes_closed': (50, 50, 150),    # Dark red
            'eyes_open': (50, 150, 50),      # Dark green
            'stationary': (200, 200, 200),   # Light gray
            'no_person': (100, 100, 100)     # Gray
        }

    def draw_skeleton(self, frame, keypoints, confidences):
        """Draw human skeleton on frame"""
        skeleton = [
            (Config.LEFT_SHOULDER, Config.RIGHT_SHOULDER),
            (Config.LEFT_SHOULDER, Config.LEFT_HIP),
            (Config.RIGHT_SHOULDER, Config.RIGHT_HIP),
            (Config.LEFT_HIP, Config.RIGHT_HIP),
            (Config.LEFT_SHOULDER, Config.LEFT_ELBOW),
            (Config.LEFT_ELBOW, Config.LEFT_WRIST),
            (Config.RIGHT_SHOULDER, Config.RIGHT_ELBOW),
            (Config.RIGHT_ELBOW, Config.RIGHT_WRIST),
            (Config.LEFT_HIP, Config.LEFT_KNEE),
            (Config.LEFT_KNEE, Config.LEFT_ANKLE),
            (Config.RIGHT_HIP, Config.RIGHT_KNEE),
            (Config.RIGHT_KNEE, Config.RIGHT_ANKLE),
            (Config.NOSE, Config.LEFT_EYE),
            (Config.NOSE, Config.RIGHT_EYE),
            (Config.LEFT_EYE, Config.LEFT_EAR),
            (Config.RIGHT_EYE, Config.RIGHT_EAR),
        ]

        for (start_idx, end_idx) in skeleton:
            if start_idx < len(confidences) and end_idx < len(confidences):
                if confidences[start_idx] > 0.4 and confidences[end_idx] > 0.4:
                    start_point = tuple(map(int, keypoints[start_idx]))
                    end_point = tuple(map(int, keypoints[end_idx]))
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

        # Draw keypoints
        keypoint_colors = {
            Config.NOSE: (0, 255, 0),
            Config.LEFT_EYE: (255, 0, 0),
            Config.RIGHT_EYE: (255, 0, 0),
            Config.LEFT_EAR: (255, 165, 0),
            Config.RIGHT_EAR: (255, 165, 0),
            Config.LEFT_SHOULDER: (0, 255, 255),
            Config.RIGHT_SHOULDER: (0, 255, 255),
            Config.LEFT_WRIST: (255, 0, 255),
            Config.RIGHT_WRIST: (255, 0, 255),
        }

        for i, (point, conf) in enumerate(zip(keypoints, confidences)):
            if conf > 0.4:
                center = tuple(map(int, point))
                color = keypoint_colors.get(i, (0, 0, 255))
                cv2.circle(frame, center, 5, color, -1)

        return frame

    def draw_eye_status(self, frame, keypoints, confidences, eyes_status):
        """Draw eye status indicator"""
        if confidences[Config.LEFT_EYE] > 0.4 and confidences[Config.RIGHT_EYE] > 0.4:
            left_eye = tuple(map(int, keypoints[Config.LEFT_EYE]))
            right_eye = tuple(map(int, keypoints[Config.RIGHT_EYE]))
            
            # Draw eye indicators
            if eyes_status == "open":
                color = (0, 255, 0)  # Green
                cv2.circle(frame, left_eye, 8, color, 2)
                cv2.circle(frame, right_eye, 8, color, 2)
            elif eyes_status == "closed":
                color = (0, 0, 255)  # Red
                cv2.line(frame, (left_eye[0]-8, left_eye[1]), (left_eye[0]+8, left_eye[1]), color, 3)
                cv2.line(frame, (right_eye[0]-8, right_eye[1]), (right_eye[0]+8, right_eye[1]), color, 3)
            else:
                color = (0, 255, 255)  # Yellow
                cv2.circle(frame, left_eye, 6, color, 1)
                cv2.circle(frame, right_eye, 6, color, 1)

        return frame

    def get_primary_status(self, movements, eyes_status):
        """Determine the primary status to display"""
        movement_names = [m[0] for m in movements]
        
        # Priority order
        if 'sleeping' in movement_names:
            return 'SLEEPING 😴', (128, 0, 128)
        
        if 'lying_down' in movement_names:
            if eyes_status == 'closed':
                return 'SLEEPING 😴', (128, 0, 128)
            return 'LYING DOWN 🛏️', (180, 100, 180)
        
        if 'both_hands_raised' in movement_names:
            return 'BOTH HANDS UP 🙌', (0, 0, 255)
        
        if 'left_hand_raised_high' in movement_names or 'right_hand_raised_high' in movement_names:
            return 'HAND RAISED HIGH ✋', (0, 0, 255)
        
        if 'left_hand_raised' in movement_names or 'right_hand_raised' in movement_names:
            return 'HAND RAISED ✋', (0, 100, 255)
        
        if 'waving' in movement_names:
            return 'WAVING 👋', (255, 0, 255)
        
        if 'running' in movement_names:
            if 'moving_left' in movement_names:
                return 'RUNNING LEFT 🏃←', (0, 255, 255)
            elif 'moving_right' in movement_names:
                return 'RUNNING RIGHT 🏃→', (0, 255, 255)
            return 'RUNNING 🏃', (0, 255, 255)
        
        if 'walking' in movement_names or 'slow_walking' in movement_names:
            if 'moving_left' in movement_names:
                return 'WALKING LEFT 🚶←', (0, 255, 0)
            elif 'moving_right' in movement_names:
                return 'WALKING RIGHT 🚶→', (0, 255, 0)
            return 'WALKING 🚶', (0, 255, 0)
        
        if 'sitting' in movement_names:
            if eyes_status == 'closed':
                return 'SITTING (DROWSY) 💤', (0, 100, 200)
            return 'SITTING 🪑', (0, 165, 255)
        
        if 'crouching' in movement_names:
            return 'CROUCHING 🧎', (100, 100, 100)
        
        if 'standing' in movement_names:
            if 'stationary' in movement_names:
                return 'STANDING STILL 🧍', (255, 255, 255)
            return 'STANDING 🧍', (255, 255, 255)
        
        if 'stationary' in movement_names:
            return 'STATIONARY 🧍', (200, 200, 200)
        
        return 'DETECTED 👤', (150, 150, 150)

    def process_frame(self, frame):
        """Process single frame"""
        processed_frame = frame.copy()
        h, w = frame.shape[:2]

        # Detect humans
        detections = self.detector.detect_humans(frame)

        # Check if no person detected
        if len(detections) == 0:
            # Draw "No Person Found" message
            self.draw_no_person_found(processed_frame)
            self.draw_statistics(processed_frame, 0)
            return processed_frame

        # Prepare tracking input
        tracking_input = []
        for det in detections:
            bbox = det['bbox']
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            tracking_input.append((bbox, centroid, det))

        # Update tracker
        tracked_objects = self.tracker.update(tracking_input)

        # Visualize results
        for object_id in tracked_objects.keys():
            info = self.tracker.get_motion_info(object_id)
            if info is None:
                continue

            det = info['detection']
            bbox = info['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Get movements and eyes status
            movements = det.get('movements', [])
            eyes_status = det.get('eyes_status', 'unknown')
            keypoints = det.get('keypoints', None)
            keypoints_conf = det.get('keypoints_conf', None)

            # Draw skeleton
            if keypoints is not None and keypoints_conf is not None:
                self.draw_skeleton(processed_frame, keypoints, keypoints_conf)
                self.draw_eye_status(processed_frame, keypoints, keypoints_conf, eyes_status)

            # Get primary status
            primary_status, primary_color = self.get_primary_status(movements, eyes_status)

            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), primary_color, 3)

            # Draw main label
            label = f"ID:{object_id} | {primary_status}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Label background
            cv2.rectangle(processed_frame, (x1, y1-label_h-15), (x1+label_w+10, y1), primary_color, -1)
            cv2.putText(processed_frame, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Draw eyes status
            eyes_label = f"Eyes: {eyes_status.upper()}"
            eyes_color = (0, 255, 0) if eyes_status == 'open' else (0, 0, 255) if eyes_status == 'closed' else (0, 255, 255)
            cv2.putText(processed_frame, eyes_label, (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, eyes_color, 2)

            # Draw movement details
            y_offset = y2 + 40
            for i, (movement, conf) in enumerate(movements[:5]):
                if i >= 5:
                    break
                detail_label = f"• {movement.replace('_', ' ').title()} ({conf:.0%})"
                cv2.putText(processed_frame, detail_label, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 18

        # Draw statistics
        self.draw_statistics(processed_frame, len(tracked_objects))

        return processed_frame

    def draw_no_person_found(self, frame):
        """Draw 'No Person Found' message"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Main message
        message = "NO PERSON FOUND"
        (msg_w, msg_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        x = (w - msg_w) // 2
        y = (h + msg_h) // 2
        
        # Draw text with shadow
        cv2.putText(frame, message, (x+3, y+3),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
        cv2.putText(frame, message, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Subtitle
        subtitle = "Waiting for human detection..."
        (sub_w, sub_h), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        sub_x = (w - sub_w) // 2
        cv2.putText(frame, subtitle, (sub_x, y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw scanning animation
        scan_y = int((self.frame_count * 3) % h)
        cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 0), 2)

    def draw_statistics(self, frame, active_tracks):
        """Draw statistics overlay"""
        h, w = frame.shape[:2]

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (10, 10), (380, 170), (0, 255, 0), 2)

        # Title
        cv2.putText(frame, "MOTION DETECTION SYSTEM", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(frame, (20, 45), (360, 45), (0, 255, 0), 1)

        # Stats
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Active Persons: {active_tracks}",
            f"Total Detected: {self.tracker.next_object_id}",
            f"Frame: {self.frame_count}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]

        y_offset = 70
        for stat in stats:
            cv2.putText(frame, stat, (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_offset += 22

        # Detection indicators (bottom left)
        indicators = [
            ("🟢 Eyes Open", (0, 255, 0)),
            ("🔴 Eyes Closed", (0, 0, 255)),
            ("🟡 Hand Raised", (0, 255, 255)),
            ("🟣 Sleeping", (128, 0, 128))
        ]
        
        y_pos = h - 100
        cv2.rectangle(frame, (10, y_pos - 10), (180, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, y_pos - 10), (180, h - 10), (100, 100, 100), 1)
        
        for text, color in indicators:
            cv2.putText(frame, text, (20, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_pos += 20

    def run(self, source=None):
        """Run the motion detection system"""
        if source is None:
            source = Config.INPUT_SOURCE

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("❌ Error: Cannot open video source")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        # Video writer
        writer = None
        if Config.SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(Config.OUTPUT_PATH, fourcc, fps, (width, height))

        print("=" * 60)
        print("🚀 ENHANCED HUMAN MOTION DETECTION SYSTEM")
        print("=" * 60)
        print("📹 Detecting:")
        print("   • Eyes Open/Closed")
        print("   • Sleeping/Lying Down")
        print("   • Sitting/Standing")
        print("   • Walking Left/Right")
        print("   • Running")
        print("   • Hand Raising")
        print("   • No Person Detection")
        print("=" * 60)
        print("Controls: 'q'=quit, 'p'=pause, 's'=screenshot")
        print("=" * 60)

        paused = False
        last_frame = None

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠️ End of video")
                        break

                    processed_frame = self.process_frame(frame)
                    last_frame = processed_frame

                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed

                    if writer is not None:
                        writer.write(processed_frame)

                    display_frame = processed_frame
                else:
                    display_frame = last_frame.copy()
                    cv2.putText(display_frame, "⏸ PAUSED", (width//2-100, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                cv2.imshow('Human Motion Detection System', display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                elif key == ord('s'):
                    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Screenshot saved: {filename}")

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

            print(f"\n✅ Processing complete!")
            print(f"📊 Total frames: {self.frame_count}")
            print(f"👥 Total persons: {self.tracker.next_object_id}")
            print(f"⏱️ Average FPS: {self.fps:.2f}")


def main():
    system = HumanMotionDetectionSystem()
    system.run(source=0)  # 0 = webcam, or use video path


if __name__ == "__main__":
    main()