import matplotlib.pyplot as plt
import os
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from face_matching import FaceMatcher

def create_face_matching_visualization():
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    print("Initializing FaceMatcher...")
    matcher = FaceMatcher()

    img1_path = r"Dataset\test\1_0_0_20161219163425847.jpg.chip.jpg"
    img2_path = r'Dataset/test/96_1_0_20170110172637082.jpg.chip.jpg'
    # img1_path = r"C:\Users\fa715\Downloads\archive (2)\FGNET\images\057A18.JPG"
    # img2_path = r'C:\Users\fa715\Downloads\archive (2)\FGNET\images\057A16.JPG'
    result = matcher.predict(img1_path, img2_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def show_img(ax, path, title_prefix, age_key, true_age_key):
        img = cv2.imread(path)
        if img is None:
            ax.text(0.5, 0.5, "Image Not Found", ha='center', va='center')
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        pred_age = result.get(age_key)
        true_age = result.get(true_age_key)
        title = f"{title_prefix}\n"
        if true_age:
            title += f"True Age: {true_age}\n"
        if pred_age is not None:
            title += f"Predicted Age: {pred_age:.1f}"
        else:
            title += "Age Prediction Failed"
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    show_img(axes[0], img1_path, "Image 1", "age_image1", "true_age_image1")
    show_img(axes[1], img2_path, "Image 2", "age_image2", "true_age_image2")
    verified = result.get("verified", False)
    status = "MATCH (Same Person)" if verified else "NO MATCH (Different People)"
    color = "green" if verified else "red"
    distance = result.get('distance', 0)
    threshold = result.get('threshold', 0)
    plt.suptitle(f"Face Verification Result: {status}\nDistance: {distance:.4f} (Threshold: {threshold})", fontsize=16, color=color, weight='bold')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "face_matching_test2.png")
    plt.savefig(save_path)
    print(f"Face matching visualization saved to {save_path}")

if __name__ == "__main__":
    create_face_matching_visualization()
