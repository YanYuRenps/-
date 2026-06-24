"""
Add signal strength annotations to search_3d_windy.png using PIL.
"""
from PIL import Image, ImageDraw, ImageFont
import os

img_path = "assets/trajectories_3d/search_3d_windy.png"
out_path = "assets/trajectories_3d/search_3d_windy_annotated.png"

img = Image.open(img_path)
draw = ImageDraw.Draw(img)

# Use a default font (PIL built-in)
try:
    font = ImageFont.truetype("arial.ttf", 16)
    font_small = ImageFont.truetype("arial.ttf", 13)
except:
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Approximate pixel coordinates for key trajectory points in the image
# Based on visual inspection of search_3d_windy.png (512x384 approx)
# UAV 0 left boundary low point (signal=0.25)
draw.text((45, 280), "signal=0.25", fill="blue", font=font_small)
# UAV 0 recovery mid point (signal=0.38)
draw.text((110, 220), "signal=0.38", fill="blue", font=font_small)
# UAV 1 boundary point (signal low)
draw.text((360, 260), "signal≈0.18", fill="darkorange", font=font_small)
# UAV 2 corner point (signal low)
draw.text((380, 330), "signal≈0.18", fill="green", font=font_small)
# Target near top
draw.text((220, 140), "Target", fill="red", font=font)

# Add a legend box for signal annotations
legend_x, legend_y = 10, 10
legend_w, legend_h = 180, 90
draw.rectangle([legend_x, legend_y, legend_x+legend_w, legend_y+legend_h],
               fill="white", outline="black", width=1)
draw.text((legend_x+8, legend_y+8), "Signal Strength", fill="black", font=font)
draw.text((legend_x+8, legend_y+30), "UAV 0: 0.25 → 0.38", fill="blue", font=font_small)
draw.text((legend_x+8, legend_y+50), "UAV 1: ≈0.18", fill="darkorange", font=font_small)
draw.text((legend_x+8, legend_y+68), "UAV 2: ≈0.18", fill="green", font=font_small)

img.save(out_path)
print(f"[VIS] Saved annotated wind trajectory: {out_path}")
