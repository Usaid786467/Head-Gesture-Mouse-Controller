"""
Facial Expression OS Navigator - Gesture Recognition Module
Advanced gesture detection with temporal smoothing and false positive reduction
"""

import numpy as np
from typing import Dict, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import time

from config import (
    gesture_thresholds,
    landmark_indices,
    adaptive_config
)
from face_processor import (
    FaceData,
    get_eye_aspect_ratio,
    get_mouth_aspect_ratio,
    calculate_distance
)


@dataclass
class GestureState:
    """Current state of all detected gestures"""
    # Eye gestures
    left_wink: bool = False
    right_wink: bool = False
    both_eyes_closed: bool = False
    
    # Mouth gestures
    mouth_open: bool = False
    smile: bool = False
    
    # Cheek gestures
    cheeks_puffed: bool = False
    
    # Head movements
    head_tilt_left: bool = False
    head_tilt_right: bool = False
    head_nod_up: bool = False
    head_nod_down: bool = False
    head_shake_left: bool = False
    head_shake_right: bool = False
    
    # Eyebrow gestures
    eyebrows_raised: bool = False
    
    # Continuous values (for mouse control)
    head_yaw: float = 0.0  # -1 (left) to 1 (right)
    head_pitch: float = 0.0  # -1 (down) to 1 (up)
    
    # Metadata
    timestamp: float = 0.0
    confidence: float = 0.0


class GestureRecognizer:
    """
    Advanced gesture recognition with temporal filtering
    Reduces false positives through frame buffering and statistical analysis
    """
    
    def __init__(self):
        """Initialize gesture recognizer with state tracking"""
        # Frame buffers for temporal consistency
        self.left_eye_buffer: Deque[float] = deque(maxlen=gesture_thresholds.EYE_AR_CONSEC_FRAMES)
        self.right_eye_buffer: Deque[float] = deque(maxlen=gesture_thresholds.EYE_AR_CONSEC_FRAMES)
        self.mouth_buffer: Deque[float] = deque(maxlen=gesture_thresholds.MOUTH_AR_CONSEC_FRAMES)
        self.cheek_buffer: Deque[float] = deque(maxlen=gesture_thresholds.CHEEK_PUFF_FRAMES)
        
        # Head pose history for movement detection
        self.head_pose_history: Deque[Tuple[float, float, float]] = deque(maxlen=10)
        
        # Baseline measurements for adaptive thresholds
        self.baseline_ear: Optional[float] = None
        self.baseline_mar: Optional[float] = None
        self.calibration_samples: int = 0
        self.calibration_complete: bool = False
        
        # State tracking
        self.current_state = GestureState()
        self.previous_state = GestureState()
        
        # Gesture cooldowns (prevent rapid repeated triggering)
        self.last_wink_time: float = 0
        self.last_mouth_open_time: float = 0
        self.last_cheek_puff_time: float = 0
        
        print("[GestureRecognizer] Initialized")
    
    def calibrate(self, face_data: FaceData) -> bool:
        """
        Calibrate baseline measurements from neutral face
        Should be called for first 30-60 frames
        
        Returns:
            True if calibration complete
        """
        if not face_data.face_detected or self.calibration_complete:
            return self.calibration_complete
        
        # Calculate current measurements
        left_eye_landmarks = face_data.landmarks[landmark_indices.LEFT_EYE]
        right_eye_landmarks = face_data.landmarks[landmark_indices.RIGHT_EYE]
        mouth_landmarks = face_data.landmarks[landmark_indices.MOUTH_OUTER]
        
        left_ear = get_eye_aspect_ratio(left_eye_landmarks)
        right_ear = get_eye_aspect_ratio(right_eye_landmarks)
        mar = get_mouth_aspect_ratio(mouth_landmarks)
        
        # Accumulate baseline
        if self.baseline_ear is None:
            self.baseline_ear = (left_ear + right_ear) / 2
            self.baseline_mar = mar
        else:
            self.baseline_ear = 0.9 * self.baseline_ear + 0.1 * ((left_ear + right_ear) / 2)
            self.baseline_mar = 0.9 * self.baseline_mar + 0.1 * mar
        
        self.calibration_samples += 1
        
        if self.calibration_samples >= 30:
            self.calibration_complete = True
            print(f"[GestureRecognizer] Calibration complete: EAR={self.baseline_ear:.3f}, MAR={self.baseline_mar:.3f}")
            return True
        
        return False
    
    def recognize(self, face_data: FaceData) -> GestureState:
        """
        Main gesture recognition function
        
        Args:
            face_data: Processed face data with landmarks
            
        Returns:
            Current gesture state
        """
        if not face_data.face_detected:
            return GestureState(timestamp=time.time(), confidence=0.0)
        
        # Store previous state
        self.previous_state = self.current_state
        self.current_state = GestureState(timestamp=time.time(), confidence=1.0)
        
        # Recognize individual gestures
        self._recognize_eye_gestures(face_data)
        self._recognize_mouth_gestures(face_data)
        self._recognize_cheek_gestures(face_data)
        self._recognize_head_movements(face_data)
        self._recognize_eyebrow_gestures(face_data)
        
        return self.current_state
    
    def _recognize_eye_gestures(self, face_data: FaceData):
        """Detect winks and blinks"""
        left_eye_landmarks = face_data.landmarks[landmark_indices.LEFT_EYE]
        right_eye_landmarks = face_data.landmarks[landmark_indices.RIGHT_EYE]
        
        left_ear = get_eye_aspect_ratio(left_eye_landmarks)
        right_ear = get_eye_aspect_ratio(right_eye_landmarks)
        
        # Add to buffers
        self.left_eye_buffer.append(left_ear)
        self.right_eye_buffer.append(right_ear)
        
        # Use adaptive threshold if calibrated
        ear_threshold = gesture_thresholds.EYE_AR_THRESH
        if self.baseline_ear is not None:
            ear_threshold = self.baseline_ear * 0.7  # 70% of baseline
        
        # Detect left wink (right eye open, left eye closed)
        if (len(self.left_eye_buffer) == self.left_eye_buffer.maxlen and
            len(self.right_eye_buffer) == self.right_eye_buffer.maxlen):
            
            left_closed = all(ear < ear_threshold for ear in self.left_eye_buffer)
            right_open = all(ear > ear_threshold for ear in self.right_eye_buffer)
            
            if left_closed and right_open:
                self.current_state.left_wink = True
            
            # Detect right wink (left eye open, right eye closed)
            right_closed = all(ear < ear_threshold for ear in self.right_eye_buffer)
            left_open = all(ear > ear_threshold for ear in self.left_eye_buffer)
            
            if right_closed and left_open:
                self.current_state.right_wink = True
            
            # Detect both eyes closed (blink)
            if left_closed and right_closed:
                self.current_state.both_eyes_closed = True
    
    def _recognize_mouth_gestures(self, face_data: FaceData):
        """Detect mouth open and smile"""
        mouth_outer = face_data.landmarks[landmark_indices.MOUTH_OUTER]
        
        mar = get_mouth_aspect_ratio(mouth_outer)
        self.mouth_buffer.append(mar)
        
        # Use adaptive threshold if calibrated
        mouth_threshold = gesture_thresholds.MOUTH_AR_OPEN_THRESH
        if self.baseline_mar is not None:
            mouth_threshold = self.baseline_mar * 1.5  # 150% of baseline
        
        # Detect mouth open
        if (len(self.mouth_buffer) == self.mouth_buffer.maxlen and
            all(m > mouth_threshold for m in self.mouth_buffer)):
            self.current_state.mouth_open = True
        
        # Detect smile (corners of mouth raised)
        left_corner = mouth_outer[0]
        right_corner = mouth_outer[6]
        top_lip = face_data.landmarks[landmark_indices.MOUTH_TOP]
        
        left_smile_ratio = (top_lip[1] - left_corner[1]) / calculate_distance(left_corner, top_lip)
        right_smile_ratio = (top_lip[1] - right_corner[1]) / calculate_distance(right_corner, top_lip)
        
        if (left_smile_ratio < gesture_thresholds.SMILE_THRESHOLD and
            right_smile_ratio < gesture_thresholds.SMILE_THRESHOLD):
            self.current_state.smile = True
    
    def _recognize_cheek_gestures(self, face_data: FaceData):
        """Detect cheek puffing"""
        left_cheek = face_data.landmarks[landmark_indices.LEFT_CHEEK]
        right_cheek = face_data.landmarks[landmark_indices.RIGHT_CHEEK]
        nose_tip = face_data.landmarks[landmark_indices.NOSE_TIP]
        
        # Calculate cheek expansion (distance from nose)
        left_dist = calculate_distance(left_cheek, nose_tip)
        right_dist = calculate_distance(right_cheek, nose_tip)
        
        # Calculate baseline distance if not calibrated
        if self.baseline_ear is not None:
            baseline_cheek_dist = (left_dist + right_dist) / 2
            cheek_expansion = baseline_cheek_dist / ((left_dist + right_dist) / 2)
            
            self.cheek_buffer.append(cheek_expansion)
            
            if (len(self.cheek_buffer) == self.cheek_buffer.maxlen and
                all(exp > gesture_thresholds.CHEEK_PUFF_THRESHOLD for exp in self.cheek_buffer)):
                self.current_state.cheeks_puffed = True
    
    def _recognize_head_movements(self, face_data: FaceData):
        """Detect head tilt, nod, and shake"""
        if face_data.head_pose is None:
            return
        
        pitch, yaw, roll = face_data.head_pose
        self.head_pose_history.append((pitch, yaw, roll))
        
        # Normalize for mouse control (-1 to 1)
        self.current_state.head_yaw = np.clip(yaw / 30.0, -1.0, 1.0)
        self.current_state.head_pitch = np.clip(pitch / 30.0, -1.0, 1.0)
        
        # Detect head tilt (roll)
        if abs(roll) > gesture_thresholds.HEAD_TILT_THRESHOLD:
            if roll < 0:
                self.current_state.head_tilt_left = True
            else:
                self.current_state.head_tilt_right = True
        
        # Detect head nod (pitch)
        if abs(pitch) > gesture_thresholds.HEAD_NOD_THRESHOLD:
            if pitch < 0:
                self.current_state.head_nod_down = True
            else:
                self.current_state.head_nod_up = True
        
        # Detect head shake (yaw)
        if abs(yaw) > gesture_thresholds.HEAD_SHAKE_THRESHOLD:
            if yaw < 0:
                self.current_state.head_shake_left = True
            else:
                self.current_state.head_shake_right = True
    
    def _recognize_eyebrow_gestures(self, face_data: FaceData):
        """Detect eyebrow raising"""
        left_eyebrow = face_data.landmarks[landmark_indices.LEFT_EYEBROW]
        right_eyebrow = face_data.landmarks[landmark_indices.RIGHT_EYEBROW]
        left_eye = face_data.landmarks[landmark_indices.LEFT_EYE]
        right_eye = face_data.landmarks[landmark_indices.RIGHT_EYE]
        
        # Calculate distance between eyebrow and eye
        left_eb_center = np.mean(left_eyebrow, axis=0)
        right_eb_center = np.mean(right_eyebrow, axis=0)
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        left_dist = calculate_distance(left_eb_center, left_eye_center)
        right_dist = calculate_distance(right_eb_center, right_eye_center)
        
        avg_dist = (left_dist + right_dist) / 2
        
        # Compare to baseline (if available)
        if self.baseline_ear is not None:
            # Approximate baseline eyebrow distance
            baseline_eb_dist = 30.0  # This should be calibrated
            if avg_dist > baseline_eb_dist * (1 + gesture_thresholds.EYEBROW_RAISE_THRESHOLD):
                self.current_state.eyebrows_raised = True
    
    def get_gesture_changes(self) -> Dict[str, bool]:
        """
        Get gestures that just became active (edge detection)
        Useful for triggering discrete actions
        
        Returns:
            Dictionary of gesture names and whether they just activated
        """
        changes = {}
        
        # Check for rising edges (gesture just became active)
        changes['left_wink_activated'] = (
            self.current_state.left_wink and not self.previous_state.left_wink
        )
        changes['right_wink_activated'] = (
            self.current_state.right_wink and not self.previous_state.right_wink
        )
        changes['mouth_open_activated'] = (
            self.current_state.mouth_open and not self.previous_state.mouth_open
        )
        changes['cheeks_puffed_activated'] = (
            self.current_state.cheeks_puffed and not self.previous_state.cheeks_puffed
        )
        changes['smile_activated'] = (
            self.current_state.smile and not self.previous_state.smile
        )
        changes['eyebrows_raised_activated'] = (
            self.current_state.eyebrows_raised and not self.previous_state.eyebrows_raised
        )
        
        return changes
    
    def reset_buffers(self):
        """Clear all temporal buffers"""
        self.left_eye_buffer.clear()
        self.right_eye_buffer.clear()
        self.mouth_buffer.clear()
        self.cheek_buffer.clear()
        self.head_pose_history.clear()
    
    def get_debug_info(self) -> Dict[str, any]:
        """Get current state for debugging"""
        return {
            'calibration_complete': self.calibration_complete,
            'calibration_samples': self.calibration_samples,
            'baseline_ear': self.baseline_ear,
            'baseline_mar': self.baseline_mar,
            'current_state': self.current_state,
            'buffer_sizes': {
                'left_eye': len(self.left_eye_buffer),
                'right_eye': len(self.right_eye_buffer),
                'mouth': len(self.mouth_buffer),
                'cheek': len(self.cheek_buffer),
                'head_pose': len(self.head_pose_history)
            }
        }


class GestureFilter:
    """
    Post-processing filter to reduce false positives
    Uses statistical methods and temporal consistency
    """
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Number of frames for moving average
        """
        self.window_size = window_size
        self.gesture_history: Deque[GestureState] = deque(maxlen=window_size)
    
    def filter(self, gesture_state: GestureState) -> GestureState:
        """
        Apply temporal filtering to reduce noise
        
        Args:
            gesture_state: Raw gesture state
            
        Returns:
            Filtered gesture state
        """
        self.gesture_history.append(gesture_state)
        
        if len(self.gesture_history) < self.window_size:
            return gesture_state
        
        # Use majority voting for boolean gestures
        filtered_state = GestureState(
            timestamp=gesture_state.timestamp,
            confidence=gesture_state.confidence
        )
        
        # Count occurrences in history
        left_wink_count = sum(1 for g in self.gesture_history if g.left_wink)
        right_wink_count = sum(1 for g in self.gesture_history if g.right_wink)
        mouth_open_count = sum(1 for g in self.gesture_history if g.mouth_open)
        
        # Apply majority voting (at least 60% of frames)
        threshold = self.window_size * 0.6
        filtered_state.left_wink = left_wink_count >= threshold
        filtered_state.right_wink = right_wink_count >= threshold
        filtered_state.mouth_open = mouth_open_count >= threshold
        
        # Average continuous values
        filtered_state.head_yaw = np.mean([g.head_yaw for g in self.gesture_history])
        filtered_state.head_pitch = np.mean([g.head_pitch for g in self.gesture_history])
        
        return filtered_state