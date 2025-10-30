"""
Facial Expression OS Navigator - Face Processing Module
High-performance facial landmark detection with MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List
import time
from dataclasses import dataclass

from config import (
    mediapipe_config, 
    camera_config, 
    landmark_indices,
    performance_config
)


@dataclass
class FaceData:
    """Container for processed face data"""
    landmarks: np.ndarray
    landmarks_3d: np.ndarray
    face_detected: bool
    confidence: float
    timestamp: float
    frame_number: int
    
    # Computed features
    head_pose: Optional[Tuple[float, float, float]] = None  # pitch, yaw, roll
    face_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


class FaceProcessor:
    """
    High-performance face detection and landmark extraction
    Uses MediaPipe Face Mesh for real-time processing
    """
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh with optimized settings"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=mediapipe_config.MAX_NUM_FACES,
            refine_landmarks=mediapipe_config.REFINE_LANDMARKS,
            min_detection_confidence=mediapipe_config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=mediapipe_config.MIN_TRACKING_CONFIDENCE
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Performance tracking
        self.frame_count = 0
        self.last_face_data: Optional[FaceData] = None
        self.processing_times: List[float] = []
        
        # Camera calibration matrix (for head pose estimation)
        self.camera_matrix = self._get_camera_matrix()
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        print("[FaceProcessor] Initialized with MediaPipe Face Mesh")
        
    def _get_camera_matrix(self) -> np.ndarray:
        """Generate camera intrinsic matrix"""
        focal_length = camera_config.FRAME_WIDTH
        center = (camera_config.FRAME_WIDTH / 2, camera_config.FRAME_HEIGHT / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        return camera_matrix
    
    def process_frame(self, frame: np.ndarray) -> FaceData:
        """
        Process a single frame and extract facial landmarks
        
        Args:
            frame: BGR image from camera
            
        Returns:
            FaceData object with landmarks and metadata
        """
        start_time = time.time()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Performance optimization
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        
        # Extract landmarks
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]  # Use first face
            
            # Convert to numpy array
            h, w = frame.shape[:2]
            landmarks_2d = np.array([
                [lm.x * w, lm.y * h] 
                for lm in face_landmarks.landmark
            ], dtype=np.float32)
            
            landmarks_3d = np.array([
                [lm.x * w, lm.y * h, lm.z * w] 
                for lm in face_landmarks.landmark
            ], dtype=np.float32)
            
            # Calculate face bounding box
            x_coords = landmarks_2d[:, 0]
            y_coords = landmarks_2d[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            face_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Estimate head pose
            head_pose = self._estimate_head_pose(landmarks_3d, frame.shape)
            
            face_data = FaceData(
                landmarks=landmarks_2d,
                landmarks_3d=landmarks_3d,
                face_detected=True,
                confidence=1.0,  # MediaPipe doesn't provide per-frame confidence
                timestamp=time.time(),
                frame_number=self.frame_count,
                head_pose=head_pose,
                face_rect=face_rect
            )
        else:
            # No face detected
            face_data = FaceData(
                landmarks=np.array([]),
                landmarks_3d=np.array([]),
                face_detected=False,
                confidence=0.0,
                timestamp=time.time(),
                frame_number=self.frame_count
            )
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        self.frame_count += 1
        self.last_face_data = face_data
        
        return face_data
    
    def _estimate_head_pose(
        self, 
        landmarks_3d: np.ndarray, 
        frame_shape: Tuple[int, ...]
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (pitch, yaw, roll) using PnP algorithm
        
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # 3D model points (approximate positions in mm)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        image_points = np.array([
            landmarks_3d[landmark_indices.NOSE_TIP][:2],
            landmarks_3d[landmark_indices.CHIN][:2],
            landmarks_3d[landmark_indices.LEFT_EYE_OUTER][:2],
            landmarks_3d[landmark_indices.RIGHT_EYE_OUTER][:2],
            landmarks_3d[landmark_indices.MOUTH_LEFT][:2],
            landmarks_3d[landmark_indices.MOUTH_RIGHT][:2]
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return (0.0, 0.0, 0.0)
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Extract Euler angles
        pitch = np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))
        yaw = np.degrees(np.arctan2(-rotation_mat[2, 0], 
                        np.sqrt(rotation_mat[2, 1]**2 + rotation_mat[2, 2]**2)))
        roll = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
        
        return (pitch, yaw, roll)
    
    def draw_landmarks(
        self, 
        frame: np.ndarray, 
        face_data: FaceData,
        show_connections: bool = True
    ) -> np.ndarray:
        """
        Draw facial landmarks on frame for visualization
        
        Args:
            frame: Input frame
            face_data: Processed face data
            show_connections: Whether to draw landmark connections
            
        Returns:
            Frame with drawn landmarks
        """
        if not face_data.face_detected:
            return frame
        
        annotated_frame = frame.copy()
        
        if show_connections:
            # Draw all facial landmarks with connections
            for idx, landmark in enumerate(face_data.landmarks):
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
        else:
            # Draw only key landmarks
            key_landmarks = (
                landmark_indices.LEFT_EYE + 
                landmark_indices.RIGHT_EYE + 
                landmark_indices.MOUTH_OUTER
            )
            for idx in key_landmarks:
                x, y = int(face_data.landmarks[idx][0]), int(face_data.landmarks[idx][1])
                cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw face bounding box
        if face_data.face_rect:
            x, y, w, h = face_data.face_rect
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return annotated_frame
    
    def get_average_processing_time(self) -> float:
        """Get average frame processing time in milliseconds"""
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times) * 1000
    
    def get_fps(self) -> float:
        """Calculate current FPS based on processing times"""
        avg_time = self.get_average_processing_time() / 1000
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def cleanup(self):
        """Release resources"""
        self.face_mesh.close()
        print("[FaceProcessor] Cleaned up resources")


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(point1 - point2)


def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """
    Calculate angle at point2 formed by point1-point2-point3
    
    Returns:
        Angle in degrees
    """
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def get_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    vertical_1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def get_mouth_aspect_ratio(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for mouth open detection
    """
    vertical = calculate_distance(mouth_landmarks[3], mouth_landmarks[9])
    horizontal = calculate_distance(mouth_landmarks[0], mouth_landmarks[6])
    
    mar = vertical / horizontal
    return mar