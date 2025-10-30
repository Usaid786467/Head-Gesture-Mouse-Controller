"""
Clean F16 Navigator - Full Screen Cursor Movement with Radar System
No face box overlay - Clean radar tracking with dots
"""

import cv2
import numpy as np
import time
import sys
import math
from collections import deque
from typing import Optional, Tuple, Dict, Any
import pyautogui

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

class FaceData:
    """Container for face detection data"""
    def __init__(self):
        self.face_detected = False
        self.face_rect = None
        self.rotation = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

class GestureState:
    """Container for gesture recognition state"""
    def __init__(self):
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        self.head_roll = 0.0
        self.head_tilt_left = False
        self.head_tilt_right = False
        self.head_nod_up = False
        self.head_nod_down = False

class CleanFaceProcessor:
    """Clean face processor - no visual overlay on face"""
    
    def __init__(self):
        # Load OpenCV cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Tracking state
        self.face_center_history = deque(maxlen=15)
        self.baseline_position = None
        self.frame_count = 0
        
        print("[Face Processor] Clean mode - no face overlay")
    
    def process_frame(self, frame) -> FaceData:
        """Process frame with clean tracking (no visual overlay)"""
        face_data = FaceData()
        self.frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        if len(faces) > 0:
            face_data.face_detected = True
            
            # Use largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            face_data.face_rect = (x, y, w, h)
            
            # Calculate stable face center
            face_center = (x + w//2, y + h//2)
            self.face_center_history.append(face_center)
            
            # Set stable baseline using multiple frames
            if self.baseline_position is None and len(self.face_center_history) >= 12:
                # Use median of recent positions for most stable baseline
                recent_centers = list(self.face_center_history)[-12:]
                median_x = sorted([c[0] for c in recent_centers])[len(recent_centers)//2]
                median_y = sorted([c[1] for c in recent_centers])[len(recent_centers)//2]
                self.baseline_position = (median_x, median_y)
                print(f"[Face Processor] Stable baseline: ({median_x:.1f}, {median_y:.1f})")
            
            # Calculate movement relative to baseline
            if self.baseline_position is not None:
                # Apply strong smoothing to eliminate jitter
                if len(self.face_center_history) >= 5:
                    recent_centers = list(self.face_center_history)[-5:]
                    smooth_x = sum(center[0] for center in recent_centers) / len(recent_centers)
                    smooth_y = sum(center[1] for center in recent_centers) / len(recent_centers)
                    smoothed_center = (smooth_x, smooth_y)
                else:
                    smoothed_center = face_center
                
                # Calculate relative movement
                dx = smoothed_center[0] - self.baseline_position[0]
                dy = smoothed_center[1] - self.baseline_position[1]
                
                # Normalize movement with proper scaling
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                
                # Enhanced scaling for full screen movement
                yaw = (dx / frame_width) * 3.0    # Increased for full screen coverage
                pitch = (dy / frame_height) * 3.0  # Increased for full screen coverage
                
                # Calculate head tilt for clicking
                roll = self.calculate_head_tilt(gray[y:y+h, x:x+w])
                
                face_data.rotation = {
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll
                }
        
        return face_data
    
    def calculate_head_tilt(self, face_roi):
        """Calculate head tilt using eye detection"""
        if face_roi.size == 0:
            return 0.0
        
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5)
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            left_center = (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
            right_center = (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
            
            dy = right_center[1] - left_center[1]
            dx = right_center[0] - left_center[0]
            
            if dx != 0:
                angle = math.atan2(dy, dx)
                roll = math.degrees(angle) / 30.0  # Normalize
                return roll
        
        return 0.0

class CleanGestureRecognizer:
    """Clean gesture recognition without visual noise"""
    
    def __init__(self):
        self.calibration_samples = 0
        self.calibration_complete = False
        
        # Gesture thresholds
        self.movement_threshold = 0.05
        self.tilt_threshold = 0.3
        self.nod_threshold = 0.25
        
        # Smoothing buffers
        self.gesture_history = deque(maxlen=7)
        self.tilt_history = deque(maxlen=10)
        
        # Timing controls
        self.last_click_time = 0
        self.click_cooldown = 1.0
        
        print("[Gesture] Clean recognition system ready")
    
    def calibrate(self, face_data: FaceData) -> bool:
        """Clean calibration process"""
        if not face_data.face_detected:
            return False
        
        self.calibration_samples += 1
        
        if self.calibration_samples >= 50:  # ~1.7 seconds at 30fps
            self.calibration_complete = True
            return True
        
        return False
    
    def recognize(self, face_data: FaceData) -> GestureState:
        """Clean gesture recognition"""
        gesture_state = GestureState()
        
        if not face_data.face_detected or not self.calibration_complete:
            return gesture_state
        
        # Get movement values
        yaw = face_data.rotation['yaw']
        pitch = face_data.rotation['pitch']
        roll = face_data.rotation['roll']
        
        # Apply smoothing
        self.gesture_history.append((yaw, pitch, roll))
        self.tilt_history.append(roll)
        
        if len(self.gesture_history) >= 5:
            recent = list(self.gesture_history)[-5:]
            gesture_state.head_yaw = sum(g[0] for g in recent) / len(recent)
            gesture_state.head_pitch = sum(g[1] for g in recent) / len(recent)
            gesture_state.head_roll = sum(g[2] for g in recent) / len(recent)
        else:
            gesture_state.head_yaw = yaw
            gesture_state.head_pitch = pitch
            gesture_state.head_roll = roll
        
        # Tilt detection for clicking
        if len(self.tilt_history) >= 7:
            recent_tilts = list(self.tilt_history)[-7:]
            avg_tilt = sum(recent_tilts) / len(recent_tilts)
            
            current_time = time.time()
            
            if avg_tilt < -self.tilt_threshold and current_time - self.last_click_time > self.click_cooldown:
                gesture_state.head_tilt_left = True
                self.last_click_time = current_time
            elif avg_tilt > self.tilt_threshold and current_time - self.last_click_time > self.click_cooldown:
                gesture_state.head_tilt_right = True
                self.last_click_time = current_time
        
        # Nod detection
        if abs(gesture_state.head_pitch) > self.nod_threshold:
            if gesture_state.head_pitch > self.nod_threshold:
                gesture_state.head_nod_down = True
            elif gesture_state.head_pitch < -self.nod_threshold:
                gesture_state.head_nod_up = True
        
        return gesture_state
    
    def reset_buffers(self):
        """Reset all buffers"""
        self.gesture_history.clear()
        self.tilt_history.clear()
        self.calibration_samples = 0
        self.calibration_complete = False

class FullScreenOSController:
    """Full screen cursor controller"""
    
    def __init__(self):
        self.mouse_enabled = False
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Start at screen center
        self.current_x = self.screen_width // 2
        self.current_y = self.screen_height // 2
        
        print(f"[OS Controller] Full screen: {self.screen_width}x{self.screen_height}")
    
    def update_mouse_position(self, yaw_delta: float, pitch_delta: float):
        """Full screen cursor movement"""
        if not self.mouse_enabled:
            return
        
        # FULL SCREEN MAPPING
        # Map head movement to full screen coordinates
        screen_center_x = self.screen_width // 2
        screen_center_y = self.screen_height // 2
        
        # Use full screen range with proper scaling
        max_movement_x = self.screen_width * 0.45   # 45% from center = 90% total range
        max_movement_y = self.screen_height * 0.45  # 45% from center = 90% total range
        
        # Calculate target position
        target_x = screen_center_x + (yaw_delta * max_movement_x)
        target_y = screen_center_y + (pitch_delta * max_movement_y)
        
        # Apply screen boundaries with small margin
        margin = 5
        target_x = max(margin, min(self.screen_width - margin, target_x))
        target_y = max(margin, min(self.screen_height - margin, target_y))
        
        # Smooth movement with less aggressive smoothing for better responsiveness
        smoothing = 0.5  # Reduced for more responsive movement
        self.current_x = self.current_x * smoothing + target_x * (1 - smoothing)
        self.current_y = self.current_y * smoothing + target_y * (1 - smoothing)
        
        # Move cursor
        try:
            pyautogui.moveTo(int(self.current_x), int(self.current_y), duration=0)
        except:
            pass
    
    def left_click(self) -> bool:
        """Left click"""
        if not self.mouse_enabled:
            return False
        try:
            pyautogui.click(button='left')
            return True
        except:
            return False
    
    def scroll(self, amount: int) -> bool:
        """Scroll"""
        if not self.mouse_enabled:
            return False
        try:
            pyautogui.scroll(amount * 3)
            return True
        except:
            return False
    
    def get_current_position(self) -> Tuple[int, int]:
        """Get current cursor position"""
        return (int(self.current_x), int(self.current_y))

class CleanF16Navigator:
    """Clean F16 Navigator with radar system"""
    
    def __init__(self):
        print("=" * 60)
        print("  F16 CLEAN NAVIGATOR")
        print("  Full Screen + Radar System")
        print("=" * 60)
        
        # Initialize components
        self.camera: Optional[cv2.VideoCapture] = None
        self.face_processor = CleanFaceProcessor()
        self.gesture_recognizer = CleanGestureRecognizer()
        self.os_controller = FullScreenOSController()
        
        # Application state
        self.running = False
        self.paused = False
        self.calibrating = True
        
        # UI state
        self.current_action = "CALIBRATING"
        self.last_success_action = "NO ACTIONS"
        self.last_operation_time = 0
        
        # Movement tracking
        self.is_moving = False
        self.movement_magnitude = 0.0
        self.movement_trail = deque(maxlen=20)  # For radar trail effect
        
        print("[F16] Clean navigator ready - no face overlay!")
    
    def initialize_camera(self) -> bool:
        """Initialize camera"""
        for camera_index in [0, 1, 2, -1]:
            try:
                self.camera = cv2.VideoCapture(camera_index)
                if self.camera.isOpened():
                    ret, test_frame = self.camera.read()
                    if ret:
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.camera.set(cv2.CAP_PROP_FPS, 30)
                        print(f"[Camera] Ready at index {camera_index}")
                        return True
                    else:
                        self.camera.release()
            except:
                pass
        
        print("[Camera] ERROR: No camera found")
        return False
    
    def process_gestures(self, face_data: FaceData):
        """Process gestures cleanly"""
        if not face_data.face_detected:
            self.current_action = "NO FACE"
            self.is_moving = False
            return
        
        # Get gesture state
        gesture_state = self.gesture_recognizer.recognize(face_data)
        
        self.current_action = "READY"
        self.is_moving = False
        
        # Calculate movement magnitude
        self.movement_magnitude = abs(gesture_state.head_yaw) + abs(gesture_state.head_pitch)
        
        # Head tilt left = Click
        if gesture_state.head_tilt_left:
            if self.os_controller.mouse_enabled and not self.paused:
                if self.os_controller.left_click():
                    self.current_action = "CLICK"
                    self.last_success_action = f"CLICK {time.strftime('%H:%M:%S')}"
                    self.last_operation_time = time.time()
        
        # Cursor movement
        if self.os_controller.mouse_enabled and not self.paused:
            movement_threshold = 0.03  # Lower threshold for more responsive movement
            
            if self.movement_magnitude > movement_threshold:
                self.is_moving = True
                
                # Direct movement mapping
                self.os_controller.update_mouse_position(
                    gesture_state.head_yaw, 
                    gesture_state.head_pitch
                )
                
                # Add to movement trail for radar
                self.movement_trail.append((gesture_state.head_yaw, gesture_state.head_pitch))
                
                # Show movement feedback
                if abs(gesture_state.head_yaw) > abs(gesture_state.head_pitch):
                    direction = "LEFT" if gesture_state.head_yaw < 0 else "RIGHT"
                    self.current_action = f"MOVE {direction}"
                else:
                    direction = "UP" if gesture_state.head_pitch < 0 else "DOWN"
                    self.current_action = f"MOVE {direction}"
        
        # Scrolling
        if gesture_state.head_nod_up or gesture_state.head_nod_down:
            if self.os_controller.mouse_enabled and not self.paused:
                scroll_amount = 1 if gesture_state.head_nod_up else -1
                if self.os_controller.scroll(scroll_amount):
                    self.current_action = f"SCROLL {'UP' if gesture_state.head_nod_up else 'DOWN'}"
                    self.last_operation_time = time.time()
    
    def draw_radar_system(self, frame, face_data):
        """Draw clean radar tracking system"""
        h, w = frame.shape[:2]
        
        # Create side panels
        left_panel = np.zeros((h, 280, 3), dtype=np.uint8)
        right_panel = np.zeros((h, 280, 3), dtype=np.uint8)
        
        # Combine frames
        result = np.zeros((h, w + 560, 3), dtype=np.uint8)
        result[:, 280:w+280] = frame  # Clean camera feed in center
        result[:, :280] = left_panel
        result[:, w+280:] = right_panel
        
        # Draw left panel - radar system
        self.draw_radar_panel(result, face_data)
        
        # Draw right panel - controls
        self.draw_control_panel(result, w)
        
        return result
    
    def draw_radar_panel(self, frame, face_data):
        """Draw radar-style tracking system"""
        # Radar circle
        radar_center = (140, 140)
        radar_radius = 80
        
        # Draw radar circles
        cv2.circle(frame, radar_center, radar_radius, (0, 100, 0), 1)
        cv2.circle(frame, radar_center, radar_radius//2, (0, 100, 0), 1)
        cv2.circle(frame, radar_center, radar_radius//4, (0, 100, 0), 1)
        
        # Draw radar crosshairs
        cv2.line(frame, 
                (radar_center[0] - radar_radius, radar_center[1]),
                (radar_center[0] + radar_radius, radar_center[1]),
                (0, 100, 0), 1)
        cv2.line(frame, 
                (radar_center[0], radar_center[1] - radar_radius),
                (radar_center[0], radar_center[1] + radar_radius),
                (0, 100, 0), 1)
        
        # Draw movement dots and trail
        if face_data.face_detected and hasattr(self, 'movement_trail'):
            # Draw movement trail
            for i, (yaw, pitch) in enumerate(self.movement_trail):
                # Convert movement to radar coordinates
                radar_x = radar_center[0] + int(yaw * radar_radius * 0.8)
                radar_y = radar_center[1] + int(pitch * radar_radius * 0.8)
                
                # Keep within radar circle
                dx = radar_x - radar_center[0]
                dy = radar_y - radar_center[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > radar_radius * 0.9:
                    factor = (radar_radius * 0.9) / distance
                    radar_x = radar_center[0] + int(dx * factor)
                    radar_y = radar_center[1] + int(dy * factor)
                
                # Draw trail dot with fading effect
                alpha = (i + 1) / len(self.movement_trail)
                dot_color = (0, int(255 * alpha), int(100 * alpha))
                dot_size = 1 if i < len(self.movement_trail) - 3 else 2
                
                cv2.circle(frame, (radar_x, radar_y), dot_size, dot_color, -1)
            
            # Draw current position with larger dot
            if len(self.movement_trail) > 0:
                current_yaw, current_pitch = self.movement_trail[-1]
                current_x = radar_center[0] + int(current_yaw * radar_radius * 0.8)
                current_y = radar_center[1] + int(current_pitch * radar_radius * 0.8)
                
                # Keep within bounds
                dx = current_x - radar_center[0]
                dy = current_y - radar_center[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > radar_radius * 0.9:
                    factor = (radar_radius * 0.9) / distance
                    current_x = radar_center[0] + int(dx * factor)
                    current_y = radar_center[1] + int(dy * factor)
                
                # Draw current position
                cv2.circle(frame, (current_x, current_y), 4, (0, 255, 255), -1)
                cv2.circle(frame, (current_x, current_y), 6, (0, 255, 255), 1)
        
        # Radar labels
        y_offset = 250
        cv2.putText(frame, "HEAD TRACKING", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
        
        # Face detection status
        face_status = "LOCKED" if face_data.face_detected else "SEARCHING"
        face_color = (0, 255, 0) if face_data.face_detected else (0, 100, 255)
        cv2.putText(frame, f"TARGET: {face_status}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, face_color, 1)
        y_offset += 20
        
        # Movement status
        move_status = "ACTIVE" if self.is_moving else "IDLE"
        move_color = (255, 255, 0) if self.is_moving else (100, 100, 100)
        cv2.putText(frame, f"MOTION: {move_status}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, move_color, 1)
        y_offset += 20
        
        # Mouse status
        mouse_status = "ENABLED" if self.os_controller.mouse_enabled else "DISABLED"
        mouse_color = (0, 255, 0) if self.os_controller.mouse_enabled else (100, 100, 100)
        cv2.putText(frame, f"CONTROL: {mouse_status}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, mouse_color, 1)
        y_offset += 25
        
        # Current cursor position
        if self.os_controller.mouse_enabled:
            cursor_x, cursor_y = self.os_controller.get_current_position()
            cv2.putText(frame, f"CURSOR:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 15
            cv2.putText(frame, f"X: {cursor_x}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            y_offset += 15
            cv2.putText(frame, f"Y: {cursor_y}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    def draw_control_panel(self, frame, cam_width):
        """Draw control panel"""
        panel_start = cam_width + 280
        y = 30
        
        # Title
        cv2.putText(frame, "F16 CONTROL", (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 40
        
        # Current action
        action_color = (0, 255, 0) if "CLICK" in self.current_action else (255, 255, 200)
        cv2.putText(frame, f"ACTION: {self.current_action}", (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, action_color, 1)
        y += 30
        
        # System status
        status = "PAUSED" if self.paused else "ACTIVE"
        status_color = (255, 255, 0) if self.paused else (0, 255, 0)
        cv2.putText(frame, f"STATUS: {status}", (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        y += 25
        
        # Controls
        cv2.putText(frame, "CONTROLS:", (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y += 20
        
        controls = [
            "SPACE = Enable Mouse",
            "M = Disable Mouse",
            "P = Pause",
            "C = Recalibrate",
            "Q = Exit"
        ]
        
        for control in controls:
            cv2.putText(frame, control, (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 220, 255), 1)
            y += 18
        
        y += 15
        
        # Gestures
        cv2.putText(frame, "GESTURES:", (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y += 20
        
        gestures = [
            "Head Movement = Cursor",
            "Tilt Left = Click",
            "Nod Up/Down = Scroll"
        ]
        
        for gesture in gestures:
            cv2.putText(frame, gesture, (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 220, 255), 1)
            y += 18
        
        y += 20
        cv2.putText(frame, "LAST ACTION:", (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
        y += 18
        cv2.putText(frame, self.last_success_action, (panel_start + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    def handle_keyboard(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:
            return True
        
        if key == ord('q') or key == ord('Q') or key == 27:
            return False
        elif key == 32:  # Space
            self.os_controller.mouse_enabled = True
            print("[F16] ✅ FULL SCREEN mouse control enabled")
        elif key == ord('m') or key == ord('M'):
            self.os_controller.mouse_enabled = False
            print("[F16] ❌ Mouse control disabled")
        elif key == ord('p') or key == ord('P'):
            self.paused = not self.paused
            print(f"[F16] {'Paused' if self.paused else 'Resumed'}")
        elif key == ord('c') or key == ord('C'):
            self.calibrating = True
            self.gesture_recognizer.reset_buffers()
            self.face_processor.baseline_position = None
            self.face_processor.face_center_history.clear()
            print("[F16] 🔄 Recalibrating for full screen movement...")
        
        return True
    
    def run(self):
        """Main execution loop"""
        if not self.initialize_camera():
            return
        
        print("\n" + "=" * 60)
        print("  F16 CLEAN NAVIGATOR - FULL SCREEN MODE")
        print("=" * 60)
        print("\n🎯 CALIBRATING...")
        print("   • Sit comfortably and look straight ahead")
        print("   • Keep head still for calibration")
        print("   • After calibration, cursor covers FULL SCREEN")
        print("   • No face box overlay - clean radar tracking!")
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Process face (no visual overlay on face)
                face_data = self.face_processor.process_frame(frame)
                
                # Handle calibration
                if self.calibrating:
                    if self.gesture_recognizer.calibrate(face_data):
                        self.calibrating = False
                        print("\n✅ CALIBRATION COMPLETE!")
                        print("\n🚀 FULL SCREEN CURSOR CONTROL READY")
                        print("   • Move head to control cursor across entire screen")
                        print("   • Tilt head left to click")
                        print("   • Press SPACE to enable mouse control")
                
                # Process gestures
                if not self.calibrating and face_data.face_detected:
                    self.process_gestures(face_data)
                
                # Draw clean radar interface (no face overlay)
                display_frame = self.draw_radar_system(frame, face_data)
                
                # Show calibration progress
                if self.calibrating and face_data.face_detected:
                    progress = min(100, int((self.gesture_recognizer.calibration_samples / 50) * 100))
                    
                    # Draw clean progress bar
                    bar_x = display_frame.shape[1]//2 - 200
                    bar_y = 50
                    bar_width = 400
                    bar_height = 25
                    
                    # Background
                    cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (30, 30, 30), -1)
                    cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 255, 255), 2)
                    
                    # Progress fill
                    progress_width = int((progress / 100.0) * (bar_width - 4))
                    if progress_width > 0:
                        cv2.rectangle(display_frame, (bar_x + 2, bar_y + 2), 
                                     (bar_x + 2 + progress_width, bar_y + bar_height - 2), (0, 255, 255), -1)
                    
                    # Progress text
                    progress_text = f"CALIBRATING FULL SCREEN MODE... {progress}%"
                    text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = bar_x + (bar_width - text_size[0]) // 2
                    cv2.putText(display_frame, progress_text, (text_x, bar_y + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display
                cv2.imshow('F16 Clean Navigator - Full Screen Mode', display_frame)
                
                # Handle keyboard
                if not self.handle_keyboard():
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n" + "=" * 60)
        print("  F16 CLEAN NAVIGATOR SHUTDOWN")
        print("=" * 60)
        
        self.os_controller.mouse_enabled = False
        
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()
        print("\n✅ Clean navigator offline")


def main():
    """Entry point"""
    try:
        print("🚀 Starting F16 Clean Navigator...")
        print("   ✨ Features: Full screen cursor + Clean radar system")
        print("   🎯 No face overlay - Clean interface")
        print("   📡 Radar tracking with movement dots")
        
        # Check dependencies
        try:
            import cv2
            print("   ✅ OpenCV ready")
        except ImportError:
            print("   ❌ Install OpenCV: pip install opencv-python")
            return
            
        try:
            import pyautogui
            print("   ✅ PyAutoGUI ready")
        except ImportError:
            print("   ❌ Install PyAutoGUI: pip install pyautogui")
            return
        
        print("   ✅ All systems ready")
        print("\n🎬 Launching clean navigator...")
        
        navigator = CleanF16Navigator()
        navigator.run()
        
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
# """
# Facial Expression OS Navigator - Main Application
# Enterprise-grade facial control system with real-time processing
# """

# import cv2
# import numpy as np
# import time
# from typing import Optional
# import sys
# from config import (
#     camera_config,
#     ui_config,
#     safety_config,
#     action_config,
#     get_config_summary
# )
# from face_processor import FaceProcessor
# from gesture_recognizer import GestureRecognizer, GestureFilter
# from os_controller import OSController


# class FacialOSNavigator:
#     """
#     Main application class - Enterprise-grade facial OS control
#     """
    
#     def __init__(self):
#         """Initialize all system components"""
#         print("=" * 60)
#         print("  FACIAL EXPRESSION OS NAVIGATOR")
#         print("  Enterprise Edition - High Precision Control")
#         print("=" * 60)
#         print(get_config_summary())
        
#         # Initialize components
#         self.camera: Optional[cv2.VideoCapture] = None
#         self.face_processor = FaceProcessor()
#         self.gesture_recognizer = GestureRecognizer()
#         self.gesture_filter = GestureFilter(window_size=3)
#         self.os_controller = OSController()
        
#         # Application state
#         self.running = False
#         self.paused = False
#         self.calibrating = True
#         self.calibration_frames = 0
        
#         # Performance tracking
#         self.frame_count = 0
#         self.start_time = time.time()
#         self.fps_history = []
        
#         # Session tracking
#         self.session_start = time.time()
#         self.total_actions = {
#             'left_clicks': 0,
#             'right_clicks': 0,
#             'scrolls': 0,
#             'menu_opens': 0
#         }
        
#         print("[FacialOSNavigator] Initialization complete\n")
    
#     def initialize_camera(self) -> bool:
#         """
#         Initialize camera with optimal settings
        
#         Returns:
#             True if successful
#         """
#         print("[Camera] Initializing...")
        
#         self.camera = cv2.VideoCapture(camera_config.CAMERA_INDEX)
        
#         if not self.camera.isOpened():
#             print("[Camera] ERROR: Could not open camera")
#             return False
        
#         # Set camera properties
#         self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.FRAME_WIDTH)
#         self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.FRAME_HEIGHT)
#         self.camera.set(cv2.CAP_PROP_FPS, camera_config.FPS_TARGET)
        
#         # Verify settings
#         actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
#         actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
#         actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
#         print(f"[Camera] Resolution: {int(actual_width)}x{int(actual_height)}")
#         print(f"[Camera] FPS: {int(actual_fps)}")
#         print("[Camera] Ready\n")
        
#         return True
    
#     def process_gestures_to_actions(self, gesture_changes: dict):
#         """
#         Map recognized gestures to OS actions
#         This is where the magic happens!
        
#         Args:
#             gesture_changes: Dictionary of activated gestures
#         """
#         # Left Click: Mouth Open
#         if gesture_changes.get('mouth_open_activated'):
#             if self.os_controller.left_click():
#                 self.total_actions['left_clicks'] += 1
        
#         # Right Click: Right Wink
#         if gesture_changes.get('right_wink_activated'):
#             if self.os_controller.right_click():
#                 self.total_actions['right_clicks'] += 1
        
#         # Start Menu: Cheeks Puffed
#         if gesture_changes.get('cheeks_puffed_activated'):
#             if self.os_controller.open_start_menu():
#                 self.total_actions['menu_opens'] += 1
        
#         # Double Click: Smile
#         if gesture_changes.get('smile_activated'):
#             self.os_controller.double_click()
        
#         # App Switcher: Eyebrows Raised
#         if gesture_changes.get('eyebrows_raised_activated'):
#             self.os_controller.switch_application()
        
#         # Continuous actions: Head movement for mouse and scroll
#         current_state = self.gesture_recognizer.current_state
        
#         # Mouse movement via head pose
#         if abs(current_state.head_yaw) > 0.1 or abs(current_state.head_pitch) > 0.1:
#             self.os_controller.update_mouse_position(
#                 current_state.head_yaw,
#                 current_state.head_pitch
#             )
        
#         # Scrolling via head tilt
#         if current_state.head_tilt_left or current_state.head_tilt_right:
#             scroll_amount = -1 if current_state.head_tilt_left else 1
#             if self.os_controller.scroll(scroll_amount, 'horizontal'):
#                 self.total_actions['scrolls'] += 1
        
#         if current_state.head_nod_up or current_state.head_nod_down:
#             scroll_amount = 1 if current_state.head_nod_up else -1
#             if self.os_controller.scroll(scroll_amount, 'vertical'):
#                 self.total_actions['scrolls'] += 1
    
#     def draw_ui_overlay(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Draw informative UI overlay on frame
        
#         Args:
#             frame: Input frame
            
#         Returns:
#             Frame with UI overlay
#         """
#         overlay = frame.copy()
#         h, w = frame.shape[:2]
        
#         # Semi-transparent background for text
#         cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
#         cv2.addWeighted(overlay, ui_config.OVERLAY_ALPHA, frame, 1 - ui_config.OVERLAY_ALPHA, 0, frame)
        
#         y_offset = 30
#         line_height = 25
        
#         # Status
#         if self.calibrating:
#             status_text = f"CALIBRATING... {self.calibration_frames}/30"
#             status_color = ui_config.COLOR_WARNING
#         elif self.paused:
#             status_text = "PAUSED"
#             status_color = ui_config.COLOR_INACTIVE
#         else:
#             status_text = "ACTIVE"
#             status_color = ui_config.COLOR_ACTIVE
        
#         cv2.putText(frame, status_text, (20, y_offset), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
#         y_offset += line_height
        
#         # FPS
#         if ui_config.SHOW_FPS:
#             fps = self.face_processor.get_fps()
#             fps_text = f"FPS: {fps:.1f}"
#             fps_color = ui_config.COLOR_ACTIVE if fps > 20 else ui_config.COLOR_WARNING
#             cv2.putText(frame, fps_text, (20, y_offset), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
#             y_offset += line_height
        
#         # Active gestures indicator
#         if ui_config.SHOW_GESTURE_INDICATORS and not self.calibrating:
#             state = self.gesture_recognizer.current_state
#             gestures = []
            
#             if state.mouth_open:
#                 gestures.append("MOUTH OPEN")
#             if state.right_wink:
#                 gestures.append("RIGHT WINK")
#             if state.left_wink:
#                 gestures.append("LEFT WINK")
#             if state.smile:
#                 gestures.append("SMILE")
#             if state.cheeks_puffed:
#                 gestures.append("CHEEKS PUFFED")
#             if state.eyebrows_raised:
#                 gestures.append("EYEBROWS UP")
            
#             if gestures:
#                 cv2.putText(frame, "Active: " + ", ".join(gestures[:2]), (20, y_offset),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_config.COLOR_ACTIVE, 1)
#                 y_offset += line_height
        
#         # Action statistics
#         cv2.putText(frame, f"Clicks: {self.total_actions['left_clicks']} | "
#                           f"Scrolls: {self.total_actions['scrolls']}", 
#                     (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
#         y_offset += line_height
        
#         # Controls help (bottom of screen)
#         help_y = h - 80
#         cv2.rectangle(frame, (10, help_y - 10), (w - 10, h - 10), (0, 0, 0), -1)
        
#         help_texts = [
#             "Controls: [P] Pause | [C] Recalibrate | [ESC] Exit",
#             "Gestures: Mouth Open=Click | Right Wink=Right Click | Cheeks Puff=Menu",
#             "Head Movement: Tilt=Mouse | Nod=Scroll | Smile=Double Click"
#         ]
        
#         for i, text in enumerate(help_texts):
#             cv2.putText(frame, text, (20, help_y + i * 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
#         # Mouse position indicator
#         mouse_x, mouse_y = self.os_controller.get_current_position()
#         cv2.circle(frame, (w - 50, 50), 30, (0, 0, 0), -1)
#         cv2.circle(frame, (w - 50, 50), 28, (100, 100, 100), 2)
        
#         # Normalized position indicator
#         norm_x = int((mouse_x / self.os_controller.screen_width) * 50) - 25
#         norm_y = int((mouse_y / self.os_controller.screen_height) * 50) - 25
#         cv2.circle(frame, (w - 50 + norm_x, 50 + norm_y), 5, ui_config.COLOR_ACTIVE, -1)
        
#         return frame
    
#     def handle_keyboard_input(self) -> bool:
#         """
#         Handle keyboard controls
        
#         Returns:
#             False if application should exit
#         """
#         key = cv2.waitKey(1) & 0xFF
        
#         if key == 255:  # No key pressed
#             return True
        
#         # ESC - Exit
#         if key == 27:  # ESC
#             print("\n[Main] Exit requested by user")
#             return False
        
#         # P - Pause/Resume
#         elif key == ord('p') or key == ord('P'):
#             self.paused = not self.paused
#             if self.paused:
#                 self.os_controller.toggle_mouse_control(False)
#                 print("[Main] PAUSED - Press 'P' to resume")
#             else:
#                 self.os_controller.toggle_mouse_control(True)
#                 print("[Main] RESUMED")
        
#         # C - Recalibrate
#         elif key == ord('c') or key == ord('C'):
#             self.calibrating = True
#             self.calibration_frames = 0
#             self.gesture_recognizer.calibration_complete = False
#             self.gesture_recognizer.calibration_samples = 0
#             self.gesture_recognizer.reset_buffers()
#             print("[Main] Recalibration started...")
        
#         # R - Reset mouse to center
#         elif key == ord('r') or key == ord('R'):
#             self.os_controller.reset()
#             print("[Main] Mouse position reset")
        
#         # D - Toggle debug overlay
#         elif key == ord('d') or key == ord('D'):
#             ui_config.SHOW_DEBUG_OVERLAY = not ui_config.SHOW_DEBUG_OVERLAY
#             print(f"[Main] Debug overlay: {ui_config.SHOW_DEBUG_OVERLAY}")
        
#         # L - Toggle landmarks
#         elif key == ord('l') or key == ord('L'):
#             ui_config.SHOW_LANDMARKS = not ui_config.SHOW_LANDMARKS
#             print(f"[Main] Landmarks display: {ui_config.SHOW_LANDMARKS}")
        
#         return True
    
#     def run(self):
#         """Main application loop"""
#         if not self.initialize_camera():
#             return
        
#         self.running = True
#         print("\n" + "=" * 60)
#         print("  SYSTEM ACTIVE - Facial Control Enabled")
#         print("=" * 60)
#         print("\nCalibration: Look straight ahead with neutral expression...")
#         print("This will take about 1 second.\n")
        
#         try:
#             while self.running:
#                 # Read frame
#                 ret, frame = self.camera.read()
#                 if not ret:
#                     print("[Main] ERROR: Failed to read frame")
#                     break
                
#                 # Mirror frame for natural interaction
#                 frame = cv2.flip(frame, 1)
                
#                 # Process face
#                 face_data = self.face_processor.process_frame(frame)
                
#                 # Handle calibration phase
#                 if self.calibrating:
#                     if self.gesture_recognizer.calibrate(face_data):
#                         self.calibrating = False
#                         print("\n[Main] ✓ Calibration complete! System is now active.\n")
#                     else:
#                         self.calibration_frames += 1
                
#                 # Process gestures (only if calibrated and not paused)
#                 if not self.calibrating and not self.paused and face_data.face_detected:
#                     # Recognize gestures
#                     gesture_state = self.gesture_recognizer.recognize(face_data)
                    
#                     # Apply temporal filtering
#                     filtered_state = self.gesture_filter.filter(gesture_state)
                    
#                     # Get gesture activations (edge detection)
#                     gesture_changes = self.gesture_recognizer.get_gesture_changes()
                    
#                     # Map gestures to actions
#                     self.process_gestures_to_actions(gesture_changes)
                
#                 # Visualization
#                 if ui_config.SHOW_LANDMARKS and face_data.face_detected:
#                     frame = self.face_processor.draw_landmarks(
#                         frame, 
#                         face_data,
#                         show_connections=False
#                     )
                
#                 if ui_config.SHOW_DEBUG_OVERLAY:
#                     frame = self.draw_ui_overlay(frame)
                
#                 # Display frame
#                 cv2.imshow('Facial Expression OS Navigator', frame)
                
#                 # Handle keyboard input
#                 if not self.handle_keyboard_input():
#                     break
                
#                 # Update frame counter
#                 self.frame_count += 1
                
#                 # Calculate session duration
#                 if self.frame_count % 300 == 0:  # Every 10 seconds at 30fps
#                     elapsed = time.time() - self.session_start
#                     print(f"[Session] Time: {elapsed/60:.1f} min | "
#                           f"Actions: {sum(self.total_actions.values())} | "
#                           f"FPS: {self.face_processor.get_fps():.1f}")
        
#         except KeyboardInterrupt:
#             print("\n[Main] Interrupted by user")
        
#         except Exception as e:
#             print(f"\n[Main] ERROR: {e}")
#             import traceback
#             traceback.print_exc()
        
#         finally:
#             self.cleanup()
    
#     def cleanup(self):
#         """Clean up resources"""
#         print("\n" + "=" * 60)
#         print("  SHUTTING DOWN")
#         print("=" * 60)
        
#         # Print session statistics
#         elapsed = time.time() - self.session_start
#         print(f"\nSession Duration: {elapsed/60:.1f} minutes")
#         print(f"Total Frames Processed: {self.frame_count}")
#         print(f"Average FPS: {self.frame_count/elapsed:.1f}")
#         print("\nActions Performed:")
#         for action, count in self.total_actions.items():
#             print(f"  {action}: {count}")
        
#         # Release resources
#         if self.camera is not None:
#             self.camera.release()
        
#         cv2.destroyAllWindows()
#         self.face_processor.cleanup()
        
#         print("\n[Main] Cleanup complete. Goodbye!\n")


# def main():
#     """Entry point"""
#     try:
#         app = FacialOSNavigator()
#         app.run()
#     except Exception as e:
#         print(f"[FATAL] Application error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()