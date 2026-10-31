import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Load label file
df = pd.read_csv("data/raw/labels.csv")
filename = df.iloc[0]["filename"]

# Read image
img = cv2.imread(f"data/raw/{filename}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: data/raw/{filename}")

# Read mask
mask_path = f"data/annotations/{filename.replace('.png','_mask.png')}"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("Mask not found, using no mask...")
else:
    mask = (mask > 127).astype("uint8")

# Show and save overlay
plt.imshow(img, cmap='gray')
if mask is not None:
    plt.imshow(mask, cmap='Reds', alpha=0.4)

plt.title("Verification Successful")
plt.savefig("reports/figures/overlay.png")
print("Overlay saved at: reports/figures/overlay.png")
