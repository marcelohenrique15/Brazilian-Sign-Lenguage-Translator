"""
Script para extração de keypoints de 800 vídeos de sinais (LIBRAS/BSL).
Gera um CSV com coordenadas de nariz e mãos (esquerda e direita) por frame.

Estrutura do CSV:
  video_index, frame_index, word,
  nose_x, nose_y, nose_z,
  lh_x0..lh_z20  (63 colunas — 21 pontos × x/y/z),
  rh_x0..rh_z20  (63 colunas)

Uso:
  python extract_keypoints.py                         # processa data/ e salva keypoints.csv
  python extract_keypoints.py --data_dir meu/path --output resultado.csv --frames 60
"""

import re
import csv
import logging
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from typing import Any, List, Tuple
from numpy import typing as npt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detector (adaptado de src/detector.py — sem dependência do pacote src)
# ---------------------------------------------------------------------------
class Detector:
    """Detecta landmarks de mãos e pose em um único frame (modo IMAGE)."""

    def __init__(self) -> None:
        self.last_pose = np.zeros(9, dtype=np.float32)
        self.last_lh   = np.zeros(63, dtype=np.float32)
        self.last_rh   = np.zeros(63, dtype=np.float32)

        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path="models/hand_landmarker.task"
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
            )
        )

        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path="models/pose_landmarker_full.task"
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
            )
        )
        logger.info("Detector inicializado.")

    # ------------------------------------------------------------------
    # Extração de pose (índice 0 = nariz)
    # ------------------------------------------------------------------
    def _extract_pose_data(self, pose_results: Any) -> npt.NDArray[np.float32]:
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]
            data: List[float] = []
            for idx in [0, 11, 12]:          # nariz, ombro-esq, ombro-dir
                lm = landmarks[idx]
                data.extend([lm.x, lm.y, lm.z])
            self.last_pose = np.array(data, dtype=np.float32)
        return self.last_pose

    # ------------------------------------------------------------------
    # Extração de mãos
    # ------------------------------------------------------------------
    def _extract_hand_data(
        self, hand_results: Any
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        current_lh = np.zeros(63, dtype=np.float32)
        current_rh = np.zeros(63, dtype=np.float32)
        found_lh = found_rh = False

        if hand_results.hand_landmarks:
            for i, handedness in enumerate(hand_results.handedness):
                label: str = handedness[0].category_name
                points: List[float] = []
                for lm in hand_results.hand_landmarks[i]:
                    points.extend([lm.x, lm.y, lm.z])
                arr = np.array(points, dtype=np.float32)

                if label == "Left":
                    current_lh = arr
                    self.last_lh = arr
                    found_lh = True
                elif label == "Right":
                    current_rh = arr
                    self.last_rh = arr
                    found_rh = True

        return (
            current_lh if found_lh else self.last_lh,
            current_rh if found_rh else self.last_rh,
        )

    # ------------------------------------------------------------------
    # Detecção principal
    # ------------------------------------------------------------------
    def detect(
        self, frame: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Retorna (pose_data[9], lh_data[63], rh_data[63]).
        pose_data[0:3] = nariz (x, y, z).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 480))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_results = self.hand_landmarker.detect(mp_image)
        pose_results = self.pose_landmarker.detect(mp_image)

        pose_data      = self._extract_pose_data(pose_results)
        lh_data, rh_data = self._extract_hand_data(hand_results)

        return pose_data, lh_data, rh_data

    def reset_buffers(self) -> None:
        self.last_pose.fill(0)
        self.last_lh.fill(0)
        self.last_rh.fill(0)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

# Regex para capturar índice (primeiros dígitos) e nome da palavra
# Exemplos: "01AcontecerSinalizador01-1.mp4"  →  índice=01, palavra=Acontecer
_VIDEO_PATTERN = re.compile(r"^(\d+)([A-Za-zÀ-ÿ]+)Sinalizador", re.IGNORECASE)


def parse_video_name(filename: str) -> Tuple[int, str]:
    """
    Extrai (video_index, word) do nome do arquivo.
    Lança ValueError se o padrão não bater.
    """
    stem = Path(filename).stem                       # remove extensão
    m = _VIDEO_PATTERN.match(stem)
    if not m:
        raise ValueError(f"Nome de arquivo fora do padrão esperado: {filename!r}")
    video_index = int(m.group(1))
    word        = m.group(2)
    return video_index, word


def sample_frames(cap: cv2.VideoCapture, n_frames: int) -> List[npt.NDArray[np.uint8]]:
    """
    Amostra exatamente *n_frames* frames igualmente espaçados do vídeo.

    Estratégia: lê TODOS os frames sequencialmente primeiro (evita imprecisão
    de seek em codecs como H.264), depois seleciona por índice via linspace.
    Se o vídeo tiver menos frames que n_frames, o último frame é repetido.
    Garante sempre len(resultado) == n_frames.
    """
    # Leitura sequencial completa — mais confiável que CAP_PROP_POS_FRAMES
    all_frames: List[npt.NDArray[np.uint8]] = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        all_frames.append(f)

    total = len(all_frames)
    if total == 0:
        return []

    # Seleciona n_frames índices espaçados uniformemente
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    # Garante exatamente n_frames — clipa índice no último frame disponível
    return [all_frames[min(idx, total - 1)] for idx in indices]


# ---------------------------------------------------------------------------
# Cabeçalho do CSV
# ---------------------------------------------------------------------------

def build_header(n_hand_points: int = 21) -> List[str]:
    cols = ["video_index", "frame_index", "word",
            "nose_x", "nose_y", "nose_z"]

    for hand in ("lh", "rh"):
        for p in range(n_hand_points):
            for axis in ("x", "y", "z"):
                cols.append(f"{hand}_{axis}{p}")

    return cols                                       # 6 + 63 + 63 = 132 colunas


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def extract(data_dir: str, output_csv: str, n_frames: int) -> None:
    data_path   = Path(data_dir)
    video_files = sorted(data_path.glob("*.mp4"))

    if not video_files:
        logger.error(f"Nenhum .mp4 encontrado em {data_path.resolve()}")
        return

    logger.info(f"{len(video_files)} vídeos encontrados em {data_path.resolve()}")

    detector = Detector()
    header   = build_header()

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        video_counter = 0  # incrementa a cada vídeo processado com sucesso

        for video_path in video_files:
            # --- parseia nome (apenas para extrair a palavra) ---
            try:
                _, word = parse_video_name(video_path.name)
            except ValueError as e:
                logger.warning(f"Pulando arquivo: {e}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"  Não foi possível abrir: {video_path}")
                continue

            frames = sample_frames(cap, n_frames)
            cap.release()

            if not frames:
                logger.warning(f"  Sem frames em {video_path.name}")
                continue

            logger.info(f"[{video_counter:03d}] {video_path.name}  →  palavra='{word}'")

            detector.reset_buffers()

            for frame_idx, frame in enumerate(frames):
                pose_data, lh_data, rh_data = detector.detect(frame)

                # nariz = primeiros 3 valores de pose_data
                nose_x, nose_y, nose_z = pose_data[0], pose_data[1], pose_data[2]

                # Substitui arrays zerados (mão/pose não detectada) por NaN
                nose_vals = [nose_x, nose_y, nose_z]
                if all(v == 0.0 for v in nose_vals):
                    nose_vals = [float("nan")] * 3

                lh_list = lh_data.tolist()
                if all(v == 0.0 for v in lh_list):
                    lh_list = [float("nan")] * 63

                rh_list = rh_data.tolist()
                if all(v == 0.0 for v in rh_list):
                    rh_list = [float("nan")] * 63

                row = (
                    [video_counter, frame_idx, word]
                    + nose_vals
                    + lh_list
                    + rh_list
                )
                writer.writerow(row)

            video_counter += 1  # só incrementa após gravar com sucesso

    logger.info(f"CSV salvo em: {Path(output_csv).resolve()}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai keypoints de vídeos de sinais para CSV."
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Pasta com os vídeos .mp4 (padrão: data/)",
    )
    parser.add_argument(
        "--output",
        default="keypoints.csv",
        help="Caminho do CSV de saída (padrão: keypoints.csv)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Número de frames amostrados por vídeo (padrão: 60)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(
        data_dir   = args.data_dir,
        output_csv = args.output,
        n_frames   = args.frames,
    )