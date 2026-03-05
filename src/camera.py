import cv2
import logging
from typing import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Camera:
    """
    Classe responsável pela interface com o hardware de vídeo.
    Gerencia capturas em alta resolução e redimensionamento para modelos de ML.
    """
    def __init__(self, camId: int = 0) -> None:
        # Câmera 0 por default ou path de vídeo
        self.cap = cv2.VideoCapture(camId)

        # Resoluções do frame para respectivamente processar e mostrar
        self.frame_res: Tuple[int, int] = (640, 480)
        self.cam_res: Tuple[int, int] = (1280, 720)

        # Resolução da captura da câmera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_res[1])
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not self.is_opened():
            logger.error("Falha ao inicializar a captura de vídeo.")
            raise RuntimeError("Não foi possível abrir a câmera.")
        
        actual_w: float = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h: float = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Câmera iniciada. Resolução real: {actual_w}x{actual_h}")

    def is_opened(self) -> bool:
        """Verifica se a câmera está funcionando."""
        return self.cap.isOpened()
    
    def get_frame(self) -> Optional[Any]:
        """Captura o frame e redimensiona para ser processado pelo modelo de ML."""
        ret, frame = self.cap.read()

        # Verifica se houve captura de frame
        if not ret:
            logger.warning("Falha na captura do frame.")
            return None
        
        frame = cv2.resize(frame, self.frame_res, interpolation=cv2.INTER_AREA)
        return frame
    
    def show_frame(self, frame: Any, title: str) -> None:
        """Abre a janela e mostra a câmera."""
        if frame is not None:
            display_frame = cv2.resize(frame, self.cam_res, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(title, display_frame)
    
    def release(self) -> None:
        """Fecha a câmera e libera os recursos alocados para o processo."""
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Câmera liberada.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()