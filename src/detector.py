import logging
import cv2
import mediapipe as mp
import numpy as np
from typing import *
from numpy import typing as npt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Detector:
    """
    Classe responsável por detectar, utilizando MediaPipe, 
    todos os 21 pontos das mãos incluindo os dedos, 
    e por detectar pontos do corpo que servirão
    como ancoragem, sendo os ombros, rosto e quadris. 
    """
    def __init__(self) -> None:
        self.last_pose = np.zeros(9, dtype=np.float32)
        self.last_lh = np.zeros(63, dtype=np.float32)
        self.last_rh = np.zeros(63, dtype=np.float32)
        
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path='models/hand_landmarker.task'),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5
            )
        )
        logger.info("Detector de Mãos inicializado")

        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path='models/pose_landmarker_full.task'),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5
            )
        )
        logger.info("Detector de Pose inicializado")

    def _extract_pose_data(self, pose_results: Any) -> npt.NDArray[np.float32]:
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]
            
            indexes = [0, 11, 12]
            data: List[float] = []
            for idx in indexes:
                lm = landmarks[idx]
                data.extend([lm.x, lm.y, lm.z])
            
            self.last_pose = np.array(data, dtype=np.float32)
        
        return self.last_pose

    def _extract_hand_data(self, hand_results: Any) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        current_lh = np.zeros(63, dtype=np.float32)
        current_rh = np.zeros(63, dtype=np.float32)
        found_lh, found_rh = False, False

        if hand_results.hand_landmarks:
            for i, handedness in enumerate(hand_results.handedness):
                label: str = handedness[0].category_name

                points = []
                for lm in hand_results.hand_landmarks[i]:
                    points.extend([lm.x, lm.y, lm.z])
                
                points_array = np.array(points, dtype=np.float32)
                
                if label == "Left":
                    current_lh = points_array
                    self.last_lh = points_array
                    found_lh = True
                
                elif label == "Right":
                    current_rh = points_array
                    self.last_rh = points_array
                    found_rh = True
        
        if found_lh:
            final_result_lh = current_lh
        else:
            final_result_lh = self.last_lh
        
        if found_rh:
            final_result_rh = current_rh
        else:
            final_result_rh = self.last_rh

        return final_result_lh, final_result_rh
    
    def detect(self, frame: npt.NDArray[np.uint8]) -> Tuple[Dict[str, Any], npt.NDArray[np.float32]]:
        detection_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 480))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detection_frame)

        hand_results = self.hand_landmarker.detect(mp_image)
        pose_results = self.pose_landmarker.detect(mp_image)

        pose_data = self._extract_pose_data(pose_results)
        lh_data, rh_data = self._extract_hand_data(hand_results)

        keypoints = np.concatenate([pose_data, lh_data, rh_data]).astype(np.float32)

        return {"hands": hand_results, "pose": pose_results}, keypoints
    
    def reset_buffers(self) -> None:
        self.last_pose.fill(0)
        self.last_lh.fill(0)
        self.last_rh.fill(0)