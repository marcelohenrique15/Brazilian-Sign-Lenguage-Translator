import time
import threading
import numpy as np
import numpy.typing as npt
from typing import *
from src.camera import Camera
from src.detector import Detector

class Thread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.running: bool = False
    
    def stop(self) -> None:
        self.running = False

class CameraThread(Thread):
    def __init__(self, cam_id = 0) -> None:
        super().__init__()
        self.camera = Camera(camId=cam_id)
        self.frame = None
        self.lock = threading.Lock()

    def run(self) -> None:
        self.running = True

        while self.running:
            new_frame = self.camera.get_frame()

            if new_frame is not None:
                with self.lock:
                    self.frame = new_frame
            
            time.sleep(0.01)
        
        self.camera.release()
    
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            
            return None

class DetectionThread(Thread):
    def __init__(self, camera_thread: CameraThread):
        super().__init__()
        self.camera_thread = camera_thread
        self.detector = Detector()

        self.results = None
        self.keypoints = None
        self.lock = threading.Lock()

    def run(self) -> None:
        self.running = True

        while self.running:
            frame = self.camera_thread.get_frame()

            if frame is not None:
                res, kp = self.detector.detect(frame)

                with self.lock:
                    self.results = res
                    self.keypoints = kp
            
            time.sleep(0.01)
    
    def get_data(self) -> Tuple[Optional[Dict[str, Any]], npt.NDArray[np.float32]]:
        with self.lock:
            return self.results, self.keypoints