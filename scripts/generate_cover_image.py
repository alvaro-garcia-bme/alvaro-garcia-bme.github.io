import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
projects_dir = os.path.join(base_dir, "public", "images", "projects")

# Load real sagittal master scan
src_path = os.path.join(projects_dir, "real_mri_sagittal.jpg")
img = Image.open(src_path).convert('RGB')

# img is 1024x1024 or 512x512
# The knee joint is centered around (0.45*w, 0.45*h)
w, h = img.size

# Crop to cinematic 4:3 or 16:10 aspect ratio centered on the joint
crop_w = int(w * 0.88)
crop_h = int(crop_w * 0.75) # 4:3 ratio
cx = int(w * 0.48)
cy = int(h * 0.46)

left = max(0, cx - crop_w // 2)
top = max(0, cy - crop_h // 2)
right = min(w, left + crop_w)
bottom = min(h, top + crop_h)

cropped = img.crop((left, top, right, bottom))
cropped = cropped.resize((1000, 750), Image.Resampling.LANCZOS)

# Enhance contrast slightly for deep blacks and crisp anatomical details
enhancer = ImageEnhance.Contrast(cropped)
enhanced = enhancer.enhance(1.15)

# Save as cover
out_path = os.path.join(projects_dir, "rsna-knee-cover.jpg")
enhanced.save(out_path, "JPEG", quality=95)
print("Saved clean realistic cover image:", out_path)
