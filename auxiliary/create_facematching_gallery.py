import os
import glob
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from auxiliary.face_matching import FaceMatcher

def create_gallery():
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    matcher = FaceMatcher()
    test_dir = os.path.join("Dataset", "test")
    images = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    if len(images) < 4:
        print("Not enough images in Dataset/test")
        return
    match_pair = None
    mismatch_pair = None
    max_try = min(25, len(images))
    for i in range(max_try):
        for j in range(i + 1, max_try):
            res = matcher.predict(images[i], images[j])
            if res.get("verified", False) and match_pair is None:
                match_pair = (images[i], images[j], res)
                break
        if match_pair:
            break
    for i in range(max_try - 1, 0, -1):
        res = matcher.predict(images[i - 1], images[i])
        if not res.get("verified", False):
            mismatch_pair = (images[i - 1], images[i], res)
            break
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    def show(ax, path, title, age_key, true_age_key, res):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        if img is None:
            ax.text(0.5, 0.5, "Image Not Found", ha="center", va="center")
        else:
            ax.imshow(img)
            pred_age = res.get(age_key)
            true_age = res.get(true_age_key)
            t = title
            if true_age is not None:
                t += f"\nTrue: {true_age}"
            if pred_age is not None:
                t += f"\nPred: {pred_age:.1f}"
            ax.set_title(t, fontsize=11)
        ax.axis("off")
    suptitle = ""
    if match_pair:
        p1, p2, r = match_pair
        show(axes[0, 0], p1, "Match A", "age_image1", "true_age_image1", r)
        show(axes[0, 1], p2, "Match B", "age_image2", "true_age_image2", r)
        status = "MATCH"
        dist = r.get("distance", 0)
        thr = r.get("threshold", 0)
        suptitle = f"Face Verification: {status} | distance={dist:.4f} thr={thr}"
    else:
        suptitle = "Face Verification: No match pair found"
    if mismatch_pair:
        p1, p2, r = mismatch_pair
        show(axes[1, 0], p1, "Non-Match A", "age_image1", "true_age_image1", r)
        show(axes[1, 1], p2, "Non-Match B", "age_image2", "true_age_image2", r)
        status2 = "NO MATCH"
        dist2 = r.get("distance", 0)
        thr2 = r.get("threshold", 0)
        suptitle += f"\nSecond Pair: {status2} | distance={dist2:.4f} thr={thr2}"
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "face_matching_gallery.png")
    plt.savefig(save_path)
    print(f"Saved gallery to {save_path}")

if __name__ == "__main__":
    create_gallery()
