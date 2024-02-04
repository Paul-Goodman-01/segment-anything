import os
import glob
import json
import random
import traceback
import tkinter as tk
from tkinter.filedialog import askopenfilename
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_fill_holes
import DenseposeGroupingColours as dgc
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

##########################################################################################################
## Custom exception class 
class ScriptException(Exception):
    pass

##########################################################################################################

# Check that required directories exist
def checkDirectories(base, dens, json, out):
    return os.path.exists(base) and os.path.exists(dens) and os.path.exists(json) and os.path.exists(out)

def doImageCheckingAndRescaling(base, dens, metrics):
    OK = False
    # Get the shapes of the two images
    base_shape = base.shape
    dens_shape = dens.shape
    json_shape = (metrics['IM_SHAPE'][0], metrics['IM_SHAPE'][1], 3)
    json_normalised = metrics[next(iter(grouping))]['NORMALISED']
    print(f"Base image shape     : {base_shape}")
    print(f"Densepose mask shape : {dens_shape}")
    print(f"JSON metrics shape   : {json_shape}")
    print(f"Normalised metrics?  : {json_normalised}")
    # OK the dimensions of the two images, and the dimensions stored in the json should all match
    # if not, rescale the densepose file to match the base image, and rescale the body metrics in
    # the json file - fail if this is not possible.
    if json_normalised == False:
        if not base_shape == dens_shape:
            print("Image shapes do not match, and mask metrics cannot be rescaled!")
        else: 
            print("Images are of correct shape, and metrics do not need to be rescaled")
            OK = True
    else:
        print("Resizing densepose mask image to match base image, and rescaling mask metrics")
        image_metrics = getRescaledResults( metrics, base_shape[1], base_shape[0] )
        dens_image = ski.transform.resize(dens, (base_shape[0], base_shape[1]), anti_aliasing=False)
        dens_shape = dens_image.shape
        json_shape = (image_metrics['IM_SHAPE'][0], image_metrics['IM_SHAPE'][1], 3)
        json_normalised = image_metrics[next(iter(grouping))]['NORMALISED']
        print(f"Base image shape     : {base_shape}")
        print(f"Densepose mask shape : {dens_shape}")
        print(f"JSON metrics shape   : {json_shape}")
        print(f"Normalised metrics?  : {json_normalised}")
        OK = True

    return OK, dens_image, image_metrics

def dumpException(e):
    print("Exception", e)
    stack_trace = traceback.format_exc()
    print("Stack Trace:")
    print(stack_trace)

# Simple function to print elements of a list
def dumpFileList(file_list):
    for i, file in enumerate(file_list):
        print(f"{i}. {file}")

def generateSAMArrays(body_part, 
                      image_metrics, 
                      area_thresh=None, 
                      area_ratio=None, 
                      max_block=None, 
                      bbox_padding=[-1,-1,-1,-1],
                      bbox_auto=[0.1, 0.1], 
                      verbose=False):
    print(f"Body part : {body_part}")
    points = bboxes = labels = []   
    if body_part in image_metrics.keys():
        metrics = image_metrics[body_part] 
        max_x = image_metrics['IM_SHAPE'][1] 
        max_y = image_metrics['IM_SHAPE'][0]
               
        # Generate area, point, box and label lists
        # Annoyingly SAM wants points in (x,y) format, but ski has them in (y,x) format
        areas  = [a for a in metrics['AREAS']]
        points = [[x, y] for y, x in metrics['CENTS']]  
        bboxes = [[x1, y1, x2, y2] for y1, x1, y2, x2 in metrics['BBOXS']]  
        labels = [1] * len(points)
        print(f" > Initial block count : {len(areas)}")

        # If 'area_thresh' is set, then only include blocks with areas
        # above that threshold. If 'area_ratio' is set, then only include blocks
        # with areas above a threshold set as that ratio of the largest area 
        #If both are set, then go with the 'area_thresh' approach
        min_area = None
        if not area_ratio == None:
            min_area = round(areas[0] / area_ratio)
        if not area_thresh == None:
            min_area = round(max_x * max_y * area_thresh * 0.01)
        idxs = getIndicesOverThreshold(areas, min_area)
        areas   = [areas[i] for i in idxs]
        bboxes  = [bboxes[i] for i in idxs]
        points  = [points[i] for i in idxs]
        labels  = [labels[i] for i in idxs]

        # If 'max_block' is set then only include that number of blocks
        if not max_block == None:
            max_block = min(max_block, len(areas))
            areas  = areas[:max_block]
            points = points[:max_block]
            labels = labels[:max_block]

        # Get a simgle bounding box from all remaining bounding boxes        
        bboxes = [getCompositeBoundingBox(bboxes)]
        if len(bboxes[0])>0: 
            print(f"  > Max x: {max_x}, Max_y: {max_y}")
            print(f"  > Inital bbox: {bboxes}")
            #Handle automatic generation of bounding box padding
            if bbox_padding[0] == -1:
                x_pad = round(bbox_auto[0] * (bboxes[0][2] - bboxes[0][0]))
                y_pad = round(bbox_auto[1] * (bboxes[0][3] - bboxes[0][1]))
                bbox_padding = [x_pad, y_pad, x_pad, y_pad]

            bboxes[0][0] = round(max(bboxes[0][0] - bbox_padding[0], 0))
            bboxes[0][1] = round(max(bboxes[0][1] - bbox_padding[1], 0)) 
            bboxes[0][2] = round(min(bboxes[0][2] + bbox_padding[2], max_x))
            bboxes[0][3] = round(min(bboxes[0][3] + bbox_padding[3], max_y))
            print(f"  > Padding    : {bbox_padding}")
            print(f"  > Final bbox : {bboxes}")
        print(f" > Final block count : {len(areas)}")

        points = np.asarray(points)
        bboxes = np.asarray(bboxes)
        labels = np.asarray(labels)

        if verbose==True:
            print(f" > Key (Body part): '{body_part}'")
            print(f" > Centroid points: '{points}'")
            print(f" > Bounding box   : '{bboxes}'")
            print(f" > Interior labels: '{labels}'")
    else:
        print(f"WARNING! : Body part '{body_part}' not found in image metrics!")
    print("-------------")

    return points, bboxes, labels

def generateSAMCompositeBodyPartsPrediction(keys, 
                                            image_metrics, 
                                            use_bbox=False, 
                                            multimask_output=True, 
                                            fill_holes=True, 
                                            max_block=2, 
                                            area_thresh=0.5, 
                                            area_ratio=None, 
                                            bbox_padding=[-1,-1,-1,-1],
                                            bbox_auto=[0.1,0.1], 
                                            verbose=False):
    if len(keys)>0:   
        #Code to create composite mask from adding individual masks
        composite_mask = np.zeros((image_metrics['IM_SHAPE'][0], image_metrics['IM_SHAPE'][1]), dtype=bool)
        composite_mask = composite_mask[None, :, :]

        if verbose==True:
            print(f"Composite mask generation on keys {keys}")
            print(f"Composite mask shape: {composite_mask.shape}")
            print(f"Use bounding box: {use_bbox}")

        for key in keys:
            mask, _, _, _, _, _ = generateSAMSingleBodyPartPrediction(key, 
                                                                      image_metrics, 
                                                                      use_bbox=use_bbox,
                                                                      max_block=max_block,
                                                                      area_thresh=area_thresh,
                                                                      area_ratio=area_ratio,
                                                                      bbox_padding=bbox_padding,
                                                                      bbox_auto=bbox_auto,
                                                                      multimask_output=multimask_output,
                                                                      verbose=verbose)
            if len(mask)>0:
                composite_mask = np.logical_or(composite_mask, mask)    
    else:
        print("WARNING! : No body parts passed to SAM composite mask generation!")
    
    if fill_holes==True:
        composite_mask = binary_fill_holes(composite_mask)
    
    return composite_mask

def generateSAMSingleBodyPartPrediction(key, 
                                        metrics, 
                                        use_bbox=False, 
                                        max_block=2, 
                                        area_thresh=0.5, 
                                        area_ratio=None, 
                                        multimask_output = False, 
                                        bbox_padding = [-1,-1,-1,-1],
                                        bbox_auto = [0.1,0.1], 
                                        verbose=False): 
    
    points, bboxes, labels = generateSAMArrays(key,
                                               metrics, 
                                               max_block=max_block,
                                               area_thresh=area_thresh,
                                               area_ratio=area_ratio,
                                               bbox_padding=bbox_padding, 
                                               bbox_auto=bbox_auto,
                                               verbose=verbose)
    masks, scores, logits = generateSAMPrediction(points, 
                                                  bboxes, 
                                                  labels, 
                                                  use_bbox=use_bbox, 
                                                  multimask_output=multimask_output)
    if multimask_output==True and len(masks)>0:
        # Return the best scoring image out of the first two masks (the third always seems to overpredict)
        print(f"> Mask Scores: {scores}")    
        scores = scores[:2] # Only use the first two masks
        max_score = max(scores)
        max_idxs = [index for index, value in enumerate(scores) if value == max_score] # Should be rare for two masks to have the same score, but...  
        idx = max_idxs[-1] # Favour the last element in the list'
        masks = [masks[idx]]
        scores = [scores[idx]]
        logits = [logits[idx]]

    if not use_bbox:
        bboxes = []
    return masks, scores, logits, points, bboxes, labels

def generateSAMPrediction(points, 
                          bboxes, 
                          labels, 
                          use_bbox = False, 
                          multimask_output = False):
    masks = scores = logits = []
    if not use_bbox:
        if len(points)>0 and len(points)==len(labels):
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=multimask_output,
            )
    else:
        if len(points)>0 and len(points)==len(labels) and len(bboxes)>0:
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bboxes,
                multimask_output=multimask_output,
            )
    return masks, scores, logits

def getCompositeBoundingBox(bbox_list):
    result = []
    min_X = min_Y = 99999
    max_X = max_Y = -99999
    for bbox in bbox_list:
        min_Y = min(min_Y, bbox[0])
        min_X = min(min_X, bbox[1])
        max_Y = max(max_Y, bbox[2])
        max_X = max(max_X, bbox[3])
    if min_X < 99999:
        result = [min_Y, min_X, max_Y, max_X]
    return result

# Get lists of files in each input directory
def getDirectoryFileLists(base, dens, json):
    base_list = getFilesInDirectory(base, image_extensions)
    dens_list = getFilesInDirectory(dens, image_extensions)
    json_list = getFilesInDirectory(json, ["json"])

    # Quick check to see if these seen to match
    print (f"Found {len(base_list)} image files in base directory")
    print (f"Found {len(dens_list)} image files in densepose mask directory")
    print (f"Found {len(json_list)} image files in densepose json directory")
    all_equal = len(base_list) == len(dens_list) == len(json_list)
    if not all_equal:
        print("WARNING! : Unequal numbers of files in input directories!")

    return base_list, dens_list, json_list

# Get all files in a directory with extensions matching a filter list
def getFilesInDirectory(root, filters):
    file_list = []
    for ext in filters:
        pattern = f"{root}/*.{ext}"
        file_list.extend(glob.glob(pattern))
    return file_list

def getFilenameTriple(base, dens_files, json_files):
    OK = False
    dens_file = json_file = None
    if not type(base) == tuple and os.path.exists(base):
        base_name = os.path.basename(base)
        file_name, _ = os.path.splitext(base_name)
        file_name = "/"+file_name+"."
        matching_dens_files = [element for element in dens_files if file_name in element]
        matching_json_files = [element for element in json_files if file_name in element]

        if len(matching_dens_files) == 0:
            print(f"WARNING! : No matching densepose mask found for '{base_name}'!")
        elif len(matching_dens_files) > 1: 
            print(f"WARNING! : Multiple matching densepose masks found for '{base_name}'!")

        if not len(matching_json_files) == 1:
            print(f"WARNING! : No matching densepose json found for '{base_name}'!")

        OK = len(matching_dens_files) == len(matching_json_files) == 1
        if OK: 
            dens_file = matching_dens_files[0]
            json_file = matching_json_files[0]
            print(f"Base file : '{base_file}'")
            print(f"Dens_file : '{dens_file}'")
            print(f"Json_file : '{json_file}'")
    else:
       print(f"WARNING! : '{base}' does not exist!") 

    return OK, dens_file, json_file

# Get an image filename by opening a file dialog
def getImageFileByDialog():
    root = tk.Tk() 
    root.withdraw()
    image_path = askopenfilename( 
        title="Select a .PNG file",
        filetypes=[("PNG files", "*.png"),("JPG files", "*.jpg"),("All files", "*.*")]
    ) 
    return image_path

# Simple function to return the indices of a list that are above a numeric threshold
def getIndicesOverThreshold(values, threshold):
    result = [i for i, value in enumerate(values) if value >= threshold]
    return result 

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

def initSegmentAnythingModel(model, checkpoint, device):
    OK = False
    # Set up segment-anything model
    print(f"Setting model '{model}' and loading weights from '{checkpoint}'")
    try:
        sam = sam_model_registry[model](checkpoint=checkpoint)
        sam.to(device=device)

        # Create predictor
        print("Creating SAM predictor...")
        predictor = SamPredictor(sam)

        OK = True
    except Exception as e:
       dumpException(e) 

    return OK, sam, predictor

def readFileTriple(base_filename, dens_filename, json_filename):
    OK = False
    base_image = dens_image = image_metrics = None
    try:
        # Load the two images 
        base_image = ski.io.imread(base_filename)
        dens_image = ski.io.imread(dens_filename)

        # Try to read in the json data 
        with open(json_filename, 'r') as file:
            image_metrics = json.load(file)

        OK = True
    except Exception as e:
        print(f"ERROR!: Failed to read inputs associated with '{base_filename}'")
        dumpException(e)

    return OK, base_image, dens_image, image_metrics

def replaceIndicesWithNames(idxs, names):
    replaced_list = [names[i] for i in idxs]
    return replaced_list

def setDirectories(in_root_dir, out_root_dir):
    base_dir = os.path.join(in_root_dir, "data")
    dens_dir = os.path.join(in_root_dir, "results_cvton") 
    json_dir = os.path.join(in_root_dir, "results_cvton_dump")
    out_dir  = os.path.join(out_root_dir, "result_masks")

    print(f"Base image directory     : '{base_dir}'")
    print(f"Densepose mask directory : '{dens_dir}'")
    print(f"Densepose json directory : '{json_dir}'")
    print(f"Output masks directory   : '{out_dir}'")

    ## Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        try: 
            os.makedirs(out_dir)
            print(f"Created output directory : '{out_dir}'")
        except:
            print(f"ERROR! : Failed to create output directory!")
    else:
        if any(os.listdir(out_dir)):
            print("WARNING! : Files in output path will be overwritten!")
 
    return base_dir, dens_dir, json_dir, out_dir 

def setSAMImage(pred, image):
    pred.set_image(image)

def showMask(mask, ax, colour=[0, 0, 255], opacity=0.6):
    print(f"Colour: {colour}, Opacity: {opacity}")
    color = np.array([colour[0]/255, colour[1]/255, colour[2]/255, opacity])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def showPoints(coords, labels, ax, marker_size=375):
    if len(coords)>0:
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def showBox(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def showBoxes(boxes, ax):
    for box in boxes:
        showBox(box, ax)

def showImageAndMask(base_image, 
                     mask, 
                     title, 
                     win_title,
                     blackout = False):
        plt.figure(figsize=(10,10))
        if not blackout:
            plt.imshow(base_image)
        else:
            black_image = np.zeros((base_image.shape[0], base_image.shape[1], 3), dtype=np.uint8)
            plt.imshow(black_image)
        showMask(mask, plt.gca(), colour=[255,255,0], opacity=0.6)
        plt.title(title, fontsize=18)
        plt.axis('on')
        fig_manager = plt.get_current_fig_manager()
        fig_manager.set_window_title(win_title)
        plt.show() 
        # Set window name
        
        print(win_title)


def showImageAndMaskDetail(key, 
                           base_image, 
                           masks, 
                           scores, 
                           points, 
                           bboxes, 
                           labels, 
                           win_title, 
                           blackout=False,
                           mask_colour=[255,255,0]):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        if not blackout:
            plt.imshow(base_image)
        else:
            mask_colour=[255,255,255]
            black_image = np.zeros((base_image.shape[0], base_image.shape[1], 3), dtype=np.uint8)
            plt.imshow(black_image)
        showMask(mask, plt.gca(), colour=mask_colour, opacity=1.0)
        showPoints(points, labels, plt.gca())
        showBoxes(bboxes, plt.gca())
        plt.title(f"{key}, Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('on')
        # Set window name
        fig_manager = plt.get_current_fig_manager()
        fig_manager.set_window_title(win_title)
        plt.show() 

##########################################################################################################

## Initialise SAM parameters
setup_sam      = True
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
sam_model      = "vit_h"
sam_device     = "cuda"
is_sam_OK      = False

if setup_sam == True:
    is_sam_OK, sam, predictor = initSegmentAnythingModel(sam_model, sam_checkpoint, sam_device)
    if not is_sam_OK:
        raise ScriptException("ERROR! : Could not initialise Segment Anything Model!")

## Initialise input and grouping dictionaries
input_mode       = "CVTON" # This sets the colours that define the masks - see 'DenseposeGroupingColours.py'
input_colours    = dgc.input_mode_dict[input_mode]
grouping_mode    = "RAW" # This sets how densepose 'fine' classifications are aggregated - see 'DenseposeGroupingColours.py'
grouping         = dgc.group_mode_dict[grouping_mode]
image_extensions = ["jpg", "png"]

use_file_dialog  = True
run_multipart    = False

##OK Let's just hardcode these directories for now 
root_dir = os.getcwd()
parent_of_root = os.path.dirname(root_dir)
in_root_dir = os.path.join(parent_of_root, "detectron2")
base_dir, dens_dir, json_dir, out_dir = setDirectories(in_root_dir, root_dir)

try:
    # Check to see if directories exist
    if not checkDirectories(base_dir, dens_dir, json_dir, out_dir):
        raise ScriptException("One or more required directories does not exist!")
    
    #OK Get file lists for each directory
    base_list, dens_list, json_list = getDirectoryFileLists(base_dir, dens_dir, json_dir)
 
    #file_idx = 1
    if not use_file_dialog: 
        file_idx = random.randint(0, len(base_list)-1)
        base_file = base_list[file_idx]
        print(f"Random file selected: Base file: {file_idx} : '{base_file}'")
    else:
        base_file = getImageFileByDialog()
        print(f"Files selected: {base_file}")
    
    OK, dens_file, json_file = getFilenameTriple(base_file, dens_list, json_list)
    if not OK:
        raise ScriptException("Matching .json or densepose file to base image failed!")

    # OK We should have the files we need now... load 'em up
    OK, base_image, dens_image, image_metrics = readFileTriple(base_file, dens_file, json_file)
    if not OK:
        raise ScriptException("Failed to read in file data!")

    OK, dens_image, image_metrics = doImageCheckingAndRescaling(base_image, dens_image, image_metrics)
    if not OK:
        raise ScriptException("Failed to rescale image or image metrics correctly!")
 
    # OK, Now we should be able to start doing something with SAM
    
    # Get the part metrics for the front torso
    #group_key="23-Head-Right"
    #group_key="2-Torso-Front"
    #group_key="9-Leg-Right-Upper-Front"
    #group_key="10-Leg-Left-Upper-Front"
    #group_key="4-Hand-Left"
    #group_key="3-Hand-Right"
   
    #key_list = ["23-Head-Right", "2-Torso-Front", "9-Leg-Right-Upper-Front", "10-Leg-Left-Upper-Front", "4-Hand-Left", "4-Hand-Left", "3-Hand-Right", "3-Hand-Right"]
    #use_bbox_list = [False, False, False, False, False, True, False, True]

    #key_list = dgc.densepose_semantic_labels
    #use_bbox_list = [True] * len(key_list)
         
    if is_sam_OK == True:
        if run_multipart==True:
            key_indices = dgc.densepose_groupings_with_hands_frontal
            for key, key_idxs in key_indices.items():
                key_list = replaceIndicesWithNames(key_idxs, dgc.densepose_semantic_labels)        
                print(f"Group key: '{key}' : '{key_list}")
                setSAMImage(predictor, base_image)
                mask = generateSAMCompositeBodyPartsPrediction(key_list, 
                                                               image_metrics, 
                                                               use_bbox=True, 
                                                               verbose=True)
                showImageAndMask(base_image, 
                                 mask, 
                                 key, 
                                 os.path.basename(base_file),
                                 blackout = False)
        else:
            #key_list = dgc.densepose_semantic_labels
            key_list = ["2-Torso-Front"]
            for keys in key_list:
                use_bbox = True
                setSAMImage(predictor, base_image)
                masks, scores, logits, points, bboxes, labels = generateSAMSingleBodyPartPrediction(keys, 
                                                                                                    image_metrics, 
                                                                                                    use_bbox, 
                                                                                                    multimask_output=True,
                                                                                                    verbose=True)
                
                points = labels = bboxes = []
                showImageAndMaskDetail(keys, 
                                       base_image, 
                                       masks, 
                                       scores, 
                                       points, 
                                       bboxes, 
                                       labels, 
                                       os.path.basename(base_file), 
                                       blackout=True)

       
except ScriptException as e:
    print("EXIT : ", e)

except Exception as e:
    dumpException(e)


