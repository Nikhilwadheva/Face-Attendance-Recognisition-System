from __future__ import annotations

import base64
import binascii
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

from database import find_employee, load_training_samples, save_employee_and_samples


class ImageInputError(ValueError):
    pass


class FaceRecognitionService:
    def __init__(self, database_path: str, face_dir: str, threshold: float = 75.0):
        self.database_path = database_path
        self.face_dir = Path(face_dir)
        self.threshold = threshold
        self._lock = threading.Lock()
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._trained = False
        self._labels: set[int] = set()
        base = Path(cv2.data.haarcascades)
        self._cascade = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        self._eyes = cv2.CascadeClassifier(str(base / "haarcascade_eye.xml"))
        if self._cascade.empty() or self._eyes.empty():
            raise RuntimeError("OpenCV could not load its face detection model.")

    @staticmethod
    def _decode_frame(value: str) -> np.ndarray:
        if not isinstance(value, str):
            raise ImageInputError("A camera image is required.")
        encoded = value.split(",", 1)[-1] if value.startswith("data:image/") else value
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageInputError("The camera image could not be decoded.") from exc
        if not raw or len(raw) > 5 * 1024 * 1024:
            raise ImageInputError("Each image must be smaller than 5 MB.")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageInputError("The uploaded file is not a readable image.")
        return image

    def _detect(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        return sorted((tuple(int(v) for v in f) for f in faces), key=lambda b: -b[2] * b[3])

    @staticmethod
    def _prep(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        crop = cv2.equalizeHist(gray[y : y + h, x : x + w])
        return cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)

    def _gray(self, value: str) -> np.ndarray:
        return cv2.cvtColor(self._decode_frame(value), cv2.COLOR_BGR2GRAY)

    def _face_crop(self, value: str) -> np.ndarray:
        gray = self._gray(value)
        faces = self._detect(gray)
        if not faces:
            raise ImageInputError("No face detected. Face the camera in good, even lighting.")
        return self._prep(gray, faces[0])

    # ---- liveness -------------------------------------------------------
    def check_liveness(self, frames: list[str]) -> bool:
        """Blink challenge: eyes must be seen open, then closed, then open again.

        A printed photo or a still image never changes state, so it fails.
        This stops casual spoofing only; it is not a certified anti-spoofing system.
        """
        states: list[bool] = []
        for frame in frames:
            gray = self._gray(frame)
            faces = self._detect(gray)
            if not faces:
                continue
            x, y, w, h = faces[0]
            upper = gray[y : y + int(h * 0.6), x : x + w]
            states.append(len(self._eyes.detectMultiScale(upper, 1.1, 6, minSize=(w // 8, w // 8))) >= 1)
        text = "".join("O" if s else "C" for s in states)
        return "OC" in text and "CO" in text[text.find("OC") :] and text.count("C") <= max(4, len(text) // 2)

    # ---- enrollment -----------------------------------------------------
    def enroll(self, employee_code: str, name: str, frames: list[str]) -> int:
        crops: list[np.ndarray] = []
        for frame in frames:
            try:
                crops.append(self._face_crop(frame))
            except ImageInputError:
                continue
        if len(crops) < 3:
            raise ImageInputError("At least 3 clear samples must contain a detectable face. Adjust lighting and try again.")

        filenames: list[str] = []
        try:
            for crop in crops:
                path = self.face_dir / f"{employee_code}_{uuid.uuid4().hex}.png"
                if not cv2.imwrite(str(path), crop):
                    raise RuntimeError("Could not save a face sample to disk.")
                filenames.append(str(path))
            old_paths = save_employee_and_samples(employee_code, name, filenames)
        except Exception:
            for filename in filenames:
                Path(filename).unlink(missing_ok=True)
            raise
        for old_path in old_paths:
            if old_path not in filenames:
                Path(old_path).unlink(missing_ok=True)
        self._trained = False
        return len(crops)

    def _train_if_needed(self) -> None:
        if self._trained:
            return
        images: list[np.ndarray] = []
        labels: list[int] = []
        for employee_id, filename in load_training_samples():
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(employee_id)
        self._labels = set(labels)
        if images:
            self._recognizer.train(images, np.asarray(labels, dtype=np.int32))
        self._trained = True

    # ---- recognition (every face in the frame) ---------------------------
    def recognize_all(self, frame: str) -> list[tuple[int, str, str, float]]:
        gray = self._gray(frame)
        faces = self._detect(gray)
        if not faces:
            raise ImageInputError("No face detected. Face the camera in good, even lighting.")
        with self._lock:
            self._train_if_needed()
            if not self._labels:
                return []
            matches: dict[int, tuple[int, str, str, float]] = {}
            for box in faces:
                label, distance = self._recognizer.predict(self._prep(gray, box))
                if distance > self.threshold or int(label) in matches:
                    continue
                employee = find_employee(int(label))
                if employee:
                    matches[int(label)] = (int(label), employee[0], employee[1], round(float(distance), 1))
        return list(matches.values())
