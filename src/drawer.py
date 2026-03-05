import cv2
import numpy as np
from typing import *

class Drawer:
    """
    Classe responsável por desenhar marcadores na câmera
    em tempo real, para vizualização de detecção e 
    dos resultados.
    """
    def __init__(self) -> None:
        self.COLOR_HAND_POINTS = (0, 255, 0)        # Verde
        self.COLOR_HAND_LINES = (255, 255, 255)     # Branco
        self.COLOR_POSE = (255, 0, 255)             # Magenta
        self.TEXT_COLOR = (88, 205, 54)             # Verde claro

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Polegar
            (0, 5), (5, 6), (6, 7), (7, 8),           # Indicador
            (5, 9), (9, 10), (10, 11), (11, 12),      # Médio
            (9, 13), (13, 14), (14, 15), (15, 16),    # Anelar
            (13, 17), (17, 18), (18, 19), (19, 20),   # Mindinho
            (0, 17)                                   # Palma
        ]

    def _draw_skeleton(self, image, points) -> None:
        for start, end in self.HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], self.COLOR_HAND_LINES, 2)
    
    def _draw_joints(self, image, points) -> None:
        for pt in points:
            cv2.circle(image, pt, 4, self.COLOR_HAND_POINTS, cv2.FILLED)
    
    def _add_hand_label(self, image, hand_data, idx, wrist_pos) -> None:
        if hasattr(hand_data, 'handedness'):
            label = hand_data.handedness[idx][0].category_name
            cv2.putText(image, label, (wrist_pos[0], wrist_pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.TEXT_COLOR, 2)
            
    def _to_pixel_coords(self, landmarks, w, h) -> List[Tuple[int, int]]:
        pixel_points = []

        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pixel_points.append((px, py))
        
        return pixel_points
    
    def draw(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Método para desenhar marcações das mãos, nariz e ombros nos frames.
        
        :param frame: Frame capturado pelo OpenCV
        :type frame: np.ndarray
        :param results: Coordenadas das landmarks detectadas pelo Detector
        :type results: Dict[str, Any]
        :return: Retorna o novo frame desenhado no formato do OpenCV para ser mostrado
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        h, w, _ = frame.shape
        annotated_image = frame.copy()

        hand_data = results.get("hands")
        if hand_data and hasattr(hand_data, 'hand_landmarks'):
            for idx, hand_lms in enumerate(hand_data.hand_landmarks):
                pixel_coords = self._to_pixel_coords(hand_lms, w, h)
                self._draw_skeleton(annotated_image, pixel_coords)
                self._draw_joints(annotated_image, pixel_coords)
                self._add_hand_label(annotated_image, hand_data, idx, pixel_coords[0])
        
        pose_data = results.get("pose")
        if pose_data and hasattr(pose_data, 'pose_landmarks'):
            for pose_lms in pose_data.pose_landmarks:
                for idx in [0, 11, 12]:
                    lm = pose_lms[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated_image, (cx, cy), 8, self.COLOR_POSE, cv2.FILLED)
                    cv2.circle(annotated_image, (cx, cy), 10, (255, 255, 255), 1)
        
        return annotated_image