import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

import tkinter as tk
from tkinter.filedialog import askopenfilename

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.9]])
        img[m] = color_mask
    ax.imshow(img)

sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

# Set-up SAM model
print("Setting model registry and loading weights...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

print("Setting mask generator...")
mask_generator = SamAutomaticMaskGenerator(sam)

# Set mask generation variables
# Call open file dialog
root = tk.Tk() 
root.withdraw()
image_path = askopenfilename()
if not image_path==None and len(image_path)>0:
    print(f"Image path set to: '{image_path}'")
else:
    sys.exit()

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (768,1024))

plt.figure(figsize=(10,10))
plt.title("Original Image")
plt.imshow(image)
plt.axis('on')
plt.show()

print("Generating masks...")
masks = mask_generator.generate(image)

#print(len(masks))
#print(masks[0].keys())

plt.figure(figsize=(10,10))
plt.title("Auto Masks")
plt.imshow(image)
show_anns(masks)
plt.axis('on')
plt.show() 