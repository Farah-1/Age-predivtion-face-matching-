import os
import cv2
import torch
import numpy as np
from deepface import DeepFace
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import AgePredictionModel
from transforms import get_val_transforms
from config import TrainConfig
from PIL import Image

class FaceMatcher:
    def __init__(self, model_path=None, device=None):
        self.cfg = TrainConfig()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is None:
            model_path = os.path.join(self.cfg.output_dir, "best_model_2nd_trial.pt")
        if not os.path.exists(model_path):
            fallback = os.path.join(self.cfg.output_dir, "best_model.pt")
            if os.path.exists(fallback):
                model_path = fallback
            else:
                print(f"Warning: Model not found at {model_path} or {fallback}")
        print(f"Loading age prediction model from {model_path}...")
        self.age_model = self._load_age_model(model_path)
        self.transform = get_val_transforms(self.cfg.img_size)
        print("Age prediction model loaded.")
    def _load_age_model(self, path):
        if not os.path.exists(path):
            model = AgePredictionModel(use_soft_labels=True, max_age=self.cfg.max_age, dropout=self.cfg.dropout, pretrained=False).to(self.device)
            return model
        ckpt = torch.load(path, map_location=self.device)
        ckpt_cfg_dict = ckpt.get("config", {})
        model = AgePredictionModel(use_soft_labels=True, max_age=ckpt_cfg_dict.get("max_age", self.cfg.max_age), dropout=ckpt_cfg_dict.get("dropout", self.cfg.dropout), pretrained=False).to(self.device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        return model
    @staticmethod
    def parse_age_from_filename(filename):
        import re
        basename = os.path.basename(filename)
        try:
            parts = basename.split('_')
            if len(parts) >= 1 and parts[0].isdigit():
                return int(parts[0])
        except:
            pass
        match = re.match(r"^\d+A(\d+)[a-zA-Z]*\.", basename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    def _predict_age_single(self, img_path):
        try:
            faces = DeepFace.extract_faces(img_path=img_path, detector_backend="retinaface", enforce_detection=False, align=True)
        except Exception as e:
            print(f"Error extracting face from {img_path}: {e}")
            return None
        if not faces:
            print(f"No face detected in {img_path}")
            return None
        face_data = faces[0]
        face_img = face_data['face']
        if face_img.dtype != np.uint8:
            if face_img.max() <= 1.0:
                face_img = (face_img * 255).astype(np.uint8)
            else:
                face_img = face_img.astype(np.uint8)
        try:
            transformed = self.transform(image=face_img)["image"]
            img_tensor = transformed.unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_age, _ = self.age_model(img_tensor)
            return pred_age.item()
        except Exception as e:
            print(f"Error during age prediction inference: {e}")
            return None
    def predict(self, img_path1, img_path2, model_name="ArcFace"):
        result = {}
        true_age1 = self.parse_age_from_filename(img_path1)
        true_age2 = self.parse_age_from_filename(img_path2)
        if true_age1 is not None:
            result["true_age_image1"] = true_age1
        if true_age2 is not None:
            result["true_age_image2"] = true_age2
        try:
            verification = DeepFace.verify(img1_path=img_path1, img2_path=img_path2, model_name=model_name, detector_backend="retinaface", align=True, enforce_detection=False)
            result.update(verification)
        except Exception as e:
            print(f"Face verification failed: {e}")
            result["error"] = str(e)
            result["verified"] = False
        age1 = self._predict_age_single(img_path1)
        age2 = self._predict_age_single(img_path2)
        result["age_image1"] = age1
        result["age_image2"] = age2
        if age1 is not None and age2 is not None:
            result["age_diff"] = abs(age1 - age2)
        return result
