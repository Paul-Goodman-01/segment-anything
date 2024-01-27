import os
import glob
import json
import traceback
import tkinter as tk
from tkinter.filedialog import askopenfilename
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import DenseposeGroupingColours as dgc
from segment_anything import sam_model_registry, SamPredictor

##########################################################################################################
## Custom exception class 
class ScriptException(Exception):
    pass

##########################################################################################################

# Get all files in a directory with extensions matching a filter list
def getFilesInDirectory(root, filters):
    file_list = []
    for ext in filters:
        pattern = f"{root}/*.{ext}"
        file_list.extend(glob.glob(pattern))
    return file_list

# Get rescaled metrics for a single body part (if necessary)
# NB: Creates a copy of the original metrics and returns the altered copy
def getRescaledMetrics(metrics, new_w, new_h):
    new_metrics = metrics
    if new_metrics['NORMALISED'] == True and new_w > 0 and new_h > 0:
        new_metrics['AREAS'] = [x * (new_w * new_h) for x in new_metrics['AREAS']]
        new_metrics['BBOXS'] = [(round(y1 * new_h), round(x1 * new_w), round(y2 * new_h), round(x2 * new_w)) for y1, x1, y2, x2 in new_metrics['BBOXS']]
        new_metrics['CENTS'] = [(round(y * new_h), round(x * new_w)) for y, x in new_metrics['CENTS']]
        new_metrics['NORMALISED'] = False
    return new_metrics

#Get rescaled results at the image level
def getRescaledResults(results, new_w, new_h):
    new_results = {}
    if new_w > 0 and new_h > 0:
        new_results = results
        for item in results['PART_LIST']:
            new_part_results = getRescaledMetrics(results[item], new_w, new_h)
            new_results[item] = new_part_results
    new_results['IM_SHAPE'] = [new_h, new_w]

    return new_results

def showMask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def showPoints(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def showBox(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

##########################################################################################################

## Set up inupt and grouping dictionaries in use
input_mode       = "CVTON" # This sets the colours that define the masks - see 'DenseposeGroupingColours.py'
input_colours    = dgc.input_mode_dict[input_mode]
grouping_mode    = "RAW" # This sets how densepose 'fine' classifications are aggregated - see 'DenseposeGroupingColours.py'
grouping         = dgc.group_mode_dict[grouping_mode]

##OK Let's just hardcode these directories for now 
root_dir = os.getcwd()
parent_of_root = os.path.dirname(root_dir)
base_dir = os.path.join(parent_of_root, "detectron2/data")
dens_dir = os.path.join(parent_of_root, "detectron2/results_cvton")
json_dir = os.path.join(parent_of_root, "detectron2/results_cvton_dump")
image_extensions = ["jpg", "png"]

setup_sam = True

if setup_sam == True:
    # Set segment-anything model variables
    sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    # Set up segment-anything model
    print("Setting model registry and loading weights...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Create predictor
    print("Creating predictor...")
    predictor = SamPredictor(sam)

# Check to see if directories exist
try:
    if not os.path.exists(base_dir) or not os.path.exists(dens_dir) or not os.path.exists(json_dir):
        raise ScriptException("One or more input directories didn't exist!")
    
    #OK Get file lists for each directory
    base_list = getFilesInDirectory(base_dir, image_extensions)
    dens_list = getFilesInDirectory(dens_dir, image_extensions)
    json_list = getFilesInDirectory(json_dir, ["json"])

    for i, file in enumerate(base_list):
        print(f"{i}. {file}")

    file_idx = 1
    base_file = os.path.basename(base_list[file_idx])
    base_name, base_ext = os.path.splitext(base_file)
    base_file = os.path.join(base_dir, base_file)
    matching_dens_files = [element for element in dens_list if base_name in element]
    matching_json_files = [element for element in json_list if base_name in element]

    print(f"Base file  : '{base_file}'")
    print(f"Dens_files : '{matching_dens_files}'")
    print(f"Json_files : '{matching_json_files}'")

    # Check to see that there is a matching json file and a densepose file
    if not len(matching_dens_files)==1 or not len(matching_json_files)==1:
        raise ScriptException("Matching .json or densepose file to base image was ambiguous!")
    else:
        json_file = matching_json_files[0]
        dens_file = matching_dens_files[0]

    # OK We should have the files we need now...
        
    # Load the two images 
    base_image = ski.io.imread(base_file)
    dens_image = ski.io.imread(dens_file)

    # Try to read in the json data 
    with open(json_file, 'r') as json_file:
        image_metrics = json.load(json_file)

    # Get the shapes of the two images
    base_shape = base_image.shape
    dens_shape = dens_image.shape
    json_shape = (image_metrics['IM_SHAPE'][0], image_metrics['IM_SHAPE'][1], 3)
    json_normalised = image_metrics[next(iter(grouping))]['NORMALISED']
    print(f"Base image shape     : {base_shape}")
    print(f"Densepose mask shape : {dens_shape}")
    print(f"JSON metrics shape   : {json_shape}")
    print(f"Normalised metrics?  : {json_normalised}")
    # OK the dimensions of the two images, and the dimensions stored in the json should all match
    # if not, rescale the densepose file to match the base image, and rescale the body metrics in
    # the json file - fail if this is not possible.
    if json_normalised == False:
        if not base_shape == dens_shape:
            raise ScriptException("Image shapes do not match, and mask metrics cannot be rescaled!")
        else: 
            print("Images are of correct shape, and metrics do not need to be rescaled")
    else:
        print("Resizing densepose mask image to match base image, and rescaling mask metrics")
        image_metrics = getRescaledResults( image_metrics, base_shape[1], base_shape[0] )
        dens_image = ski.transform.resize(dens_image, (base_shape[0], base_shape[1]), anti_aliasing=False)
        dens_shape = dens_image.shape
        json_shape = (image_metrics['IM_SHAPE'][0], image_metrics['IM_SHAPE'][1], 3)
        json_normalised = image_metrics[next(iter(grouping))]['NORMALISED']
        print(f"Base image shape     : {base_shape}")
        print(f"Densepose mask shape : {dens_shape}")
        print(f"JSON metrics shape   : {json_shape}")
        print(f"Normalised metrics?  : {json_normalised}")
        
    # OK, Now we should be able to start doing something 
    
    # Get the part metrics for the front torso
    #group_key="23-Head-Right"
    #group_key="2-Torso-Front"
    #group_key="9-Leg-Right-Upper-Front"
    #group_key="10-Leg-Left-Upper-Front"
    group_key="4-Hand-Left"
    #group_key="3-Hand-Right"

    metrics = image_metrics[group_key]
    multimask_output = False
    multi_points = False

    # Annoyingly sam wants points in (x,y) format, but ski has them in (y,x) format
    test_points = [[x, y] for y, x in metrics['CENTS']]
    test_bboxes = [[x1, y1, x2, y2] for y1, x1, y2, x2 in metrics['BBOXS']]
    test_labels = [1] * len(test_points)
    if multi_points == False:
        test_points = [test_points[0]]
        test_bboxes = [test_bboxes[0]]
        test_labels = [test_labels[0]]
    test_points = np.asarray(test_points)
    test_bboxes = np.asarray(test_bboxes)
    test_labels = np.asarray(test_labels)

    print(group_key)
    print(test_points)
    print(test_bboxes)
    print(test_labels)

    if setup_sam == True:
        predictor.set_image(base_image)

        masks, scores, logits = predictor.predict(
            point_coords=test_points,
            point_labels=test_labels,
            #box=test_bboxes,
            multimask_output=multimask_output,
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(base_image)
            showMask(mask, plt.gca())
            showPoints(test_points, test_labels, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show() 

        print(masks[0])

except ScriptException as e:
    print("EXIT : ", e)

except Exception as e:
    print("Exception", e)
    stack_trace = traceback.format_exc()
    print("Stack Trace:")
    print(stack_trace)


