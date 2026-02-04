import os
import cv2
import torch
import numpy as np
from deepface import DeepFace
from model import AgePredictionModel
from transforms import get_val_transforms
from config import TrainConfig
from PIL import Image

class FaceMatcher:
    def __init__(self, model_path=None, device=None):
        self.cfg = TrainConfig()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path is None:
            # Default to the best model in outputs directory
            model_path = os.path.join(self.cfg.output_dir, "best_model_2nd_trial.pt")
            
        if not os.path.exists(model_path):
            # Fallback to best_model.pt if 2nd_trial doesn't exist
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
            print("Model file does not exist. Initializing empty model structure.")
            model = AgePredictionModel(
                use_soft_labels=True,
                max_age=self.cfg.max_age,
                dropout=self.cfg.dropout,
                pretrained=False
            ).to(self.device)
            return model

        ckpt = torch.load(path, map_location=self.device)
        ckpt_cfg_dict = ckpt.get("config", {})
        
        model = AgePredictionModel(
            use_soft_labels=True, 
            max_age=ckpt_cfg_dict.get("max_age", self.cfg.max_age),
            dropout=ckpt_cfg_dict.get("dropout", self.cfg.dropout),
            pretrained=False
        ).to(self.device)
        
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            # Maybe it's just the state dict
            model.load_state_dict(ckpt)
            
        model.eval()
        return model

    @staticmethod
    def parse_age_from_filename(filename):
        """
        Parses the age from the filename based on known formats (UTKFace, FG-NET).
        Returns None if parsing fails.
        """
        import re
        basename = os.path.basename(filename)
        
        # 1. Try UTKFace format: [age]_[gender]_[race]_[date].jpg
        # Example: 10_0_0_20170110220255346.jpg.chip.jpg
        # Or simple: 25_1_0.jpg
        try:
            parts = basename.split('_')
            if len(parts) >= 1 and parts[0].isdigit():
                return int(parts[0])
        except:
            pass
            
        # 2. Try FG-NET format: [ID]A[AGE][suffix].JPG
        # Example: 001A02.JPG -> Age 2
        # Example: 001A43a.JPG -> Age 43
        # Regex: Starts with digits, then 'A', then digits (age), then optional letters, then .extension
        match = re.match(r"^\d+A(\d+)[a-zA-Z]*\.", basename, re.IGNORECASE)
        if match:
            return int(match.group(1))
            
        return None

    def _predict_age_single(self, img_path):
        """
        Extracts face from image and predicts age.
        """
        # Extract face using DeepFace to get the aligned face
        try:
            # extract_faces returns a list of dicts: [{'face': np.array, 'facial_area': dict, 'confidence': float}]
            # face is RGB, normalized to [0, 1] usually
            faces = DeepFace.extract_faces(
                img_path=img_path, 
                detector_backend="retinaface", # Good detection
                enforce_detection=False,
                align=True
            )
        except Exception as e:
            print(f"Error extracting face from {img_path}: {e}")
            return None

        if not faces:
            print(f"No face detected in {img_path}")
            return None
        
        # Take the first face (usually the most prominent one)
        face_data = faces[0]
        face_img = face_data['face'] # Numpy array
        
        # DeepFace usually returns [0, 1] float. 
        # Check and convert to [0, 255] uint8 for Albumentations.
        if face_img.dtype != np.uint8:
             if face_img.max() <= 1.0:
                 face_img = (face_img * 255).astype(np.uint8)
             else:
                 face_img = face_img.astype(np.uint8)
            
        # The model expects RGB. DeepFace returns RGB.
        
        # Apply transforms
        # transform expects 'image' kwarg
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
        """
        Takes 2 image paths, predicts their ages, and does face matching.
        """
        result = {}
        
        # 0. Parse True Ages (if available)
        true_age1 = self.parse_age_from_filename(img_path1)
        true_age2 = self.parse_age_from_filename(img_path2)
        
        if true_age1 is not None:
            result["true_age_image1"] = true_age1
        if true_age2 is not None:
            result["true_age_image2"] = true_age2
        
        # 1. Face Matching (Verification)
        print(f"Running face verification with {model_name}...")
        try:
            # DeepFace.verify returns dict with 'verified', 'distance', 'threshold', 'model', 'similarity_metric'
            verification = DeepFace.verify(
                img1_path=img_path1, 
                img2_path=img_path2, 
                model_name=model_name,
                detector_backend="retinaface",
                align=True,
                enforce_detection=False
            )
            result.update(verification)
        except Exception as e:
            print(f"Face verification failed: {e}")
            result["error"] = str(e)
            result["verified"] = False

        # 2. Age Prediction
        print("Predicting ages...")
        age1 = self._predict_age_single(img_path1)
        age2 = self._predict_age_single(img_path2)
        
        result["age_image1"] = age1
        result["age_image2"] = age2
        
        if age1 is not None and age2 is not None:
            result["age_diff"] = abs(age1 - age2)
        
        return result

import argparse
import glob

def run_directory_demo(matcher, dir_path, dataset_name="Dataset"):
    """
    Helper function to run a demo on the first two images found in a directory.
    """
    if not os.path.exists(dir_path):
        print(f"\nSkipping {dataset_name}: Directory not found at {dir_path}")
        return

    print(f"\n--- Testing with {dataset_name} dataset at {dir_path} ---")
    
    # Support both jpg and JPG
    images = glob.glob(os.path.join(dir_path, "*.jpg")) + glob.glob(os.path.join(dir_path, "*.JPG"))
    images = sorted(images) # Sort to ensure reproducibility
    
    if len(images) >= 2:
        img1 = images[0]
        img2 = images[1]
        print(f"Image 1: {img1}")
        print(f"Image 2: {img2}")
        
        result = matcher.predict(img1, img2)
        print(f"\n=== Result ({dataset_name}) ===")
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print(f"Not enough images in {dataset_name} directory (found {len(images)}).")

def main(img1=None, img2=None):
    # Hardcoded dataset paths
  
    # Initialize matcher
    matcher = FaceMatcher()

    result = matcher.predict(img1,img2)
    print("\n=== Result (Custom) ===")
    for k, v in result.items():
        print(f"{k}: {v}")
        

if __name__ == "__main__":
    path1 = r"C:\Users\fa715\Downloads\archive (2)\FGNET\images\057A16.JPG"
    path2 = r"C:\Users\fa715\Downloads\archive (2)\FGNET\images\057A18.JPG"
    # path1 = r"C:\Users\fa715\Downloads\archive (2)\FGNET\images\046A12.JPG"
    # path2 = r"C:\Users\fa715\Downloads\archive (2)\FGNET\images\046A17.JPG"
    main(path1, path2)  