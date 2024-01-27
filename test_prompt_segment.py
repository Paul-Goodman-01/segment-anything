import torch
import torchvision
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

import tkinter as tk
from tkinter.filedialog import askopenfilename

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Set segment-anything model variables
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

# Set mask generation variables
# Call open file dialog
root = tk.Tk() 
root.withdraw()
image_path = askopenfilename()

if not image_path==None and len(image_path)>0:
    print(f"Image path set to: '{image_path}'")
else:
    sys.exit()

# Set display variables
show_initial_image = False
show_prompt_point = True
run_multipoint_predictor = False
show_final_result = True

# Load and resize image 
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (768,1024))

# Set prompt points
input_point = np.array([[333, 119]])
input_label = np.array([1])
multi_points = np.array([[350, 500], [350, 200]])
multi_labels = np.array([1, 1])

# Show initial image
if show_initial_image==True:
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()

# Set-up SAM model
print("Setting model registry and loading weights...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Create predictor
print("Creating predictor...")
predictor = SamPredictor(sam)
predictor.set_image(image)

# Show single prompt point
if (show_prompt_point==True):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

# Run initial prediction
print(f"Running predictor on point {input_point}...")
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Run multi-prompt prediction
if run_multipoint_predictor==True:
    # Multipoint approach
    print(f"Running predictor on multi-point prompt...")
    mask_input = logits[np.argmax(scores), :, :]
    masks, scores, logits = predictor.predict(
        point_coords=multi_points,
        point_labels=multi_labels,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

print(masks.shape)  # (number_of_masks) x H x W

if show_final_result==True:
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(multi_points, multi_labels, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('on')
        plt.show()  




