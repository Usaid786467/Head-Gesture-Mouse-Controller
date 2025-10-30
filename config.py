"""
Facial Expression OS Navigator - Configuration Module
Enterprise-grade configuration management with adaptive thresholds
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class CameraConfig:
    """Camera and video processing settings"""
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_TARGET: int = 30
    
@dataclass
class MediaPipeConfig:
    """MediaPipe Face Mesh configuration"""
    MAX_NUM_FACES: int = 1
    REFINE_LANDMARKS: bool = True
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.6
    
@dataclass
class GestureThresholds:
    """Adaptive thresholds for gesture recognition"""
    # Eye aspect ratio for wink detection
    EYE_AR_THRESH: float = 0.21
    EYE_AR_CONSEC_FRAMES: int = 2
    
    # Mouth aspect ratio for click detection
    MOUTH_AR_OPEN_THRESH: float = 0.6
    MOUTH_AR_CONSEC_FRAMES: int = 3
    
    # Cheek puff detection (volume-based)
    CHEEK_PUFF_THRESHOLD: float = 1.25
    CHEEK_PUFF_FRAMES: int = 4
    
    # Head pose thresholds (in degrees)
    HEAD_TILT_THRESHOLD: float = 15.0
    HEAD_NOD_THRESHOLD: float = 12.0
    HEAD_SHAKE_THRESHOLD: float = 18.0
    
    # Eyebrow raise detection
    EYEBROW_RAISE_THRESHOLD: float = 0.15
    
    # Smile detection
    SMILE_THRESHOLD: float = 0.45
    
@dataclass
class MouseConfig:
    """Mouse control settings with smoothing"""
    SENSITIVITY: float = 2.5
    SMOOTHING_FACTOR: float = 0.35  # Lower = smoother, higher = more responsive
    DEADZONE_RADIUS: int = 5  # Pixels of no-movement zone
    ACCELERATION_CURVE: float = 1.8  # Exponential acceleration
    SCREEN_PADDING: int = 20  # Keep cursor away from edges
    
@dataclass
class ActionConfig:
    """Action triggering and cooldown settings"""
    CLICK_COOLDOWN_MS: int = 300
    SCROLL_COOLDOWN_MS: int = 150
    MENU_COOLDOWN_MS: int = 800
    DOUBLE_ACTION_THRESHOLD_MS: int = 500
    
    # Scroll parameters
    SCROLL_AMOUNT: int = 3
    SCROLL_ACCELERATION: bool = True
    
@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    ENABLE_GPU: bool = True
    FRAME_SKIP_THRESHOLD: float = 0.033  # Skip if processing takes > 33ms
    PROFILING_ENABLED: bool = False
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
@dataclass
class SafetyConfig:
    """Safety and accessibility features"""
    EMERGENCY_EXIT_KEY: str = "esc"
    PAUSE_KEY: str = "p"
    CALIBRATION_KEY: str = "c"
    
    # Auto-pause if face not detected
    AUTO_PAUSE_NO_FACE_SECONDS: float = 3.0
    
    # Fatigue detection
    ENABLE_FATIGUE_DETECTION: bool = True
    FATIGUE_WARNING_MINUTES: int = 20
    
    # Confirmation for critical actions
    REQUIRE_CONFIRMATION_CRITICAL: bool = True

@dataclass
class UIConfig:
    """User interface settings"""
    SHOW_DEBUG_OVERLAY: bool = True
    SHOW_LANDMARKS: bool = False
    SHOW_FPS: bool = True
    SHOW_GESTURE_INDICATORS: bool = True
    
    # Visual feedback colors (BGR)
    COLOR_ACTIVE: Tuple[int, int, int] = (0, 255, 0)
    COLOR_INACTIVE: Tuple[int, int, int] = (128, 128, 128)
    COLOR_WARNING: Tuple[int, int, int] = (0, 165, 255)
    COLOR_ERROR: Tuple[int, int, int] = (0, 0, 255)
    
    OVERLAY_ALPHA: float = 0.7


class AdaptiveConfig:
    """
    Dynamic configuration that adapts to user behavior and environment
    """
    def __init__(self):
        self.user_calibration: Dict[str, float] = {}
        self.environment_brightness: float = 1.0
        self.performance_mode: str = "balanced"  # "performance", "balanced", "accuracy"
        
    def calibrate_user(self, baseline_measurements: Dict[str, float]):
        """
        Personalize thresholds based on user's facial structure
        """
        self.user_calibration = baseline_measurements
        
    def adjust_for_lighting(self, brightness_level: float):
        """
        Adjust detection confidence based on lighting conditions
        """
        self.environment_brightness = brightness_level
        
    def set_performance_mode(self, mode: str):
        """
        Optimize for different priorities: performance vs accuracy
        """
        if mode not in ["performance", "balanced", "accuracy"]:
            raise ValueError("Invalid performance mode")
        self.performance_mode = mode


# Facial landmark indices (MediaPipe Face Mesh)
class LandmarkIndices:
    """Key facial landmark indices for gesture recognition"""
    
    # Eyes
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    
    # Mouth
    MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    
    # Nose (for head pose estimation)
    NOSE_TIP = 1
    NOSE_BRIDGE = 168
    
    # Eyebrows
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]
    
    # Cheeks
    LEFT_CHEEK = 205
    RIGHT_CHEEK = 425
    
    # Face oval (for head pose)
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    # Head pose reference points
    FOREHEAD = 10
    CHIN = 152
    LEFT_TEMPLE = 234
    RIGHT_TEMPLE = 454


# Global configuration instances
camera_config = CameraConfig()
mediapipe_config = MediaPipeConfig()
gesture_thresholds = GestureThresholds()
mouse_config = MouseConfig()
action_config = ActionConfig()
performance_config = PerformanceConfig()
safety_config = SafetyConfig()
ui_config = UIConfig()
adaptive_config = AdaptiveConfig()
landmark_indices = LandmarkIndices()


def get_config_summary() -> str:
    """Return a formatted summary of current configuration"""
    return f"""
=== Facial Expression OS Navigator Configuration ===
Camera: {camera_config.FRAME_WIDTH}x{camera_config.FRAME_HEIGHT} @ {camera_config.FPS_TARGET}fps
Detection Confidence: {mediapipe_config.MIN_DETECTION_CONFIDENCE}
Mouse Sensitivity: {mouse_config.SENSITIVITY}
Performance Mode: {adaptive_config.performance_mode}
Safety Features: {'Enabled' if safety_config.ENABLE_FATIGUE_DETECTION else 'Disabled'}
================================================
"""