import os
import glob
import json
import copy
import random
import traceback
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import cvton_colour_groupings as ccg
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import binary_dilation
from PIL import Image
import pauls_utils as pu
import densepose_functions as df
import skeleton_functions as skel

##########################################################################################################
## Custom exception class 
class ScriptException(Exception):
    pass

##########################################################################################################

# Adds a buffer around a binary mask area of 'buffer_size' pixels
def addBufferToMask(mask, buffer_size=5):
    buffered_mask = binary_dilation(mask, iterations=buffer_size)
    return buffered_mask

# Check that required directories exist
def checkDirectories(base, dens, json, out):
    return os.path.exists(base) and os.path.exists(dens) and os.path.exists(json) and os.path.exists(out)

# Check that required directories exist
def checkDirectories(base, dens, json, out):
    return os.path.exists(base) and os.path.exists(dens) and os.path.exists(json) and os.path.exists(out)

def countNonBlackPixels(image):
    grayscale_image = ski.color.rgb2gray(image)
    non_black_pixels = np.count_nonzero(grayscale_image)
    return non_black_pixels

def countTrueValues(array_2d):
    array_np = np.array(array_2d)
    count_true = np.sum(array_np)
    return count_true

def createBlendedImage(image1, image2, alpha=0.5):
    image1_float = ski.img_as_float(image1)
    image2_float = ski.img_as_float(image2)
    blended_image = alpha * image1_float + (1 - alpha) * image2_float
    blended_image = ski.exposure.rescale_intensity(blended_image, in_range=(0, 1))
    blended_image = ski.img_as_ubyte(blended_image)
    return blended_image

def createCompositeImage(image1, image2, mask_colour=[255,0,255]):
    image1_uint8 = ski.img_as_ubyte(image1)
    image2_uint8 = ski.img_as_ubyte(image2)
    mask, _ = getBinaryMaskWithTolerance(image2_uint8, mask_colour)
    replaced_image = np.where(mask[:, :, np.newaxis], image1_uint8, image2_uint8)
    return replaced_image

def createColouredImage(image, colour=(0, 0, 0)):
    coloured_image = np.full_like(image, colour)
    coloured_image = ski.img_as_ubyte(coloured_image)
    return coloured_image

def createMaskedImage(original_image, mask_image, back_colour='black', rep_colour=None):
    back_image = createColouredImage(original_image, colour=back_colour)
    mask_image_coloured = ski.color.gray2rgb(mask_image)
    if not rep_colour==None:    
        original_image = createColouredImage(original_image, colour=rep_colour)    
    composite_image = np.where(mask_image_coloured > 0, original_image, back_image)
    composite_image = ski.img_as_ubyte(composite_image)
    return composite_image

def createSAMinputs(create_props, image_metrics, body_parts, skeleton):
    points = []
    labels = []
    bbox   = None
    masks  = None
    #check dictionary keys
    if not create_props['AI_POINTS'] == None:
        print("Adding AI points")
        new_points, new_labels = samProcessAbsolutePoints(create_props['AI_POINTS'], True)
        points.extend(new_points)
        labels.extend(new_labels)

    if not create_props['RI_POINTS'] == None:
        print("Adding RI points")
        new_points, new_labels = samProcessRelativePoints(create_props['RI_POINTS'], image_metrics, True)
        points.extend(new_points)
        labels.extend(new_labels)

    if not create_props['AE_POINTS'] == None:
        print("Adding AE points")
        new_points, new_labels = samProcessAbsolutePoints(create_props['AE_POINTS'], False)
        points.extend(new_points)
        labels.extend(new_labels)

    if not create_props['RE_POINTS'] == None:
        print("Adding RE points")
        new_points, new_labels = samProcessRelativePoints(create_props['RE_POINTS'], image_metrics, False)
        points.extend(new_points)
        labels.extend(new_labels)

    if not create_props['SI_POINTS'] == None:
        print("Adding SI points")
        new_points, new_labels = samProcessSkeletonPoints(create_props['SI_POINTS'], skeleton, True)
        points.extend(new_points)
        labels.extend(new_labels)

    if not create_props['SE_POINTS'] == None:
        print("Adding SE points")
        new_points, new_labels = samProcessSkeletonPoints(create_props['SE_POINTS'], skeleton, False)
        points.extend(new_points)
        labels.extend(new_labels)

    if not create_props['BBOX'] == None:
        print("Adding Bounding Box")
        bbox = samProcessBoundingBox(create_props['BBOX'], body_parts)
        
    if not create_props['MASK'] == None:
        print("Adding Mask")
        masks = create_props['MASK']
    
    multi_out = create_props['MULTI_OUT'] 
    print("Set multimask output to {multi_out}")

    if not create_props['FORCE_MULTI'] == None:
        force_multi = create_props['FORCE_MULTI']
        print(f"Set forced multimask return to {force_multi}")
        
    return points, labels, bbox, masks, multi_out, force_multi

def doImageCheckingAndRescaling(base, dens, grouping, image_metrics, body_metrics, skeleton_metrics):
    OK = False
    # Get the shapes of the two images
    base_shape = base.shape
    dens_shape = dens.shape
    json_shape = (image_metrics['IM_SHAPE'][0], image_metrics['IM_SHAPE'][1], 3)
    json_normalised = body_metrics[next(iter(grouping))]['NORMALISED']
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
        new_image_metrics    = df.getRescaledImageMetrics(image_metrics, base_shape[1], base_shape[0]) 
        new_body_metrics     = df.getRescaledResults(body_metrics, base_shape[1], base_shape[0])
        new_skeleton_metrics = skel.skeletonGetRescaledBones(skeleton, base_shape[1], base_shape[0])
        dens_image = ski.transform.resize(dens, (base_shape[0], base_shape[1]), anti_aliasing=False)
        dens_image = ski.img_as_ubyte(dens_image)
        dens_shape = dens_image.shape
        json_shape = (new_image_metrics['IM_SHAPE'][0], new_image_metrics['IM_SHAPE'][1], 3)
        json_normalised = new_body_metrics[next(iter(grouping))]['NORMALISED']
                
        print(f"Base image shape     : {base_shape}")
        print(f"Densepose mask shape : {dens_shape}")
        print(f"JSON metrics shape   : {json_shape}")
        print(f"Normalised metrics?  : {json_normalised}")
        OK = True
        return OK, dens_image, new_image_metrics, new_body_metrics, new_skeleton_metrics

    return OK, dens_image, image_metrics, body_metrics, skeleton_metrics

def doPreprocessImage(image, show_image=False):
    #Get all non white pixels in the image.
    gray_image = ski.color.rgb2gray(image)
    binary_mask = gray_image > 0
    label_image = ski.measure.label(binary_mask)
    regions = ski.measure.regionprops(label_image)
    mask = np.zeros_like(gray_image, dtype=bool)
    areas = [region.area for region in regions]
    
    # Sort by descending area, get sorted indices, resort results by indices
    sorted_idxs = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
     
    # Get largest area
    mask |= label_image == regions[sorted_idxs[0]].label

    # Retain image pixels within mask
    masked_image = image.copy()
    masked_image[~mask] = 0
   
    if (show_image==True):
        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title('Original Image')
        axs[1].imshow(masked_image)
        axs[1].set_title('Masked Image')
        plt.show()

    return masked_image

# Creates a mask based on a single, specified colour
def getBinaryMask(image, colour):
    mask = np.all(image == colour, axis=-1)
    return mask, colour

# Creates a mask based on tolerance to a certain colour in linear space
def getBinaryMaskWithTolerance(image, colour, tolerance=20):
    colour_diff = np.linalg.norm(image - colour, axis=-1)
    binary_mask = colour_diff <= tolerance
    return binary_mask, colour

def getDefaultSAMInputDictionary():
    result = {}
    result['AI_POINTS'] = None
    result['RI_POINTS'] = None
    result['AE_POINTS'] = None
    result['RE_POINTS'] = None
    result['SI_POINTS'] = None
    result['SE_POINTS'] = None
    result['BBOX'] = None
    result['MASK'] = None
    result['MULTI_OUT'] = True
    result['FORCE_MULTI'] = None
    return result

# Create a composite mask based on lookup of body parts by densepose grouping dictionary key
def getGroupMask(image, group_name, mask_buffer=0, verbose = False):
    composite_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    group_idxs = grouping[group_name]  
    
    if verbose == True:
        print(f"Group name: {group_name}")
        print(f"Group idxs: {group_idxs}")
        print(composite_mask.shape)
    
    for i, idx in enumerate(group_idxs):
        if i==0:
            key_colour = input_colours[idx]

        if verbose==True:
            print(f' > processing idx: {idx} colour: {input_colours[idx]} key_colour: {key_colour}')
        mask, _ = getBinaryMaskWithTolerance(image, input_colours[idx])

        if mask_buffer > 0:
            mask = addBufferToMask(mask, mask_buffer)
        composite_mask = np.logical_or(composite_mask, mask)
    
    return composite_mask, key_colour

# Get lists of files in each input directory
def getDirectoryFileLists(base, dens, json, image_extensions):
    base_list = pu.getFilesInDirectory(base, image_extensions)
    dens_list = pu.getFilesInDirectory(dens, image_extensions)
    json_list = pu.getFilesInDirectory(json, ["json"])

    # Quick check to see if these seen to match
    print (f"Found {len(base_list)} image files in base directory")
    print (f"Found {len(dens_list)} image files in densepose mask directory")
    print (f"Found {len(json_list)} image files in densepose json directory")
    all_equal = len(base_list) == len(dens_list) == len(json_list)
    if not all_equal:
        print("WARNING! : Unequal numbers of files in input directories!")

    return base_list, dens_list, json_list

# Returns the corresponding denspose and json files for a base image
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
            print(f"Base file : '{base}'")
            print(f"Dens_file : '{dens_file}'")
            print(f"Json_file : '{json_file}'")
    else:
       print(f"WARNING! : '{base}' does not exist!") 

    return OK, dens_file, json_file

def getUsedColors(image):
    pixels = image.reshape((-1, image.shape[2]))
    unique_colors = np.unique(pixels, axis=0)
    return unique_colors

# Initialise the Segment Anything Model
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
       pu.dumpException(e) 

    return OK, sam, predictor

# Read the combination of base image, denspose image and json file
def readFileTriple(base_filename, dens_filename, json_filename):
    OK = False
    base_image = dens_image = image_metrics = None
    try:
        # Load the two images 
        base_image = ski.io.imread(base_filename)
        dens_image = ski.io.imread(dens_filename)
        dens_image = doPreprocessImage(dens_image)

        # Try to read in the json data 
        with open(json_filename, 'r') as file:
            temp_dict = json.load(file)
            image_metrics   = temp_dict['IMAGE_DATA']
            body_parts      = temp_dict['BODY_PARTS']
            skeleton        = temp_dict['SKELETON'] 
        
        OK = True
    except Exception as e:
        print(f"ERROR!: Failed to read inputs associated with '{base_filename}'")
        pu.dumpException(e)

    return OK, base_image, dens_image, image_metrics, body_parts, skeleton

def samProcessAbsolutePoints(create_data, include):
    new_points = None
    new_labels = None
    if not create_data == None and len(create_data)>0:
        new_points = copy.copy(create_data)
        new_labels = [include] * len(new_points)
    return new_points, new_labels

def samProcessRelativePoints(create_data, image_metrics, include):
    new_points = None
    new_labels = None
    if not create_data == None and len(create_data)>0:
        w = image_metrics['IM_SHAPE'][1]
        h = image_metrics['IM_SHAPE'][0]
        if not w==None and w > 0 and not h==None and h > 0: 
            new_points = [[x * w, y * h] for x, y in create_data]
            new_labels = [include] * len(new_points)
    return new_points, new_labels

def samProcessSkeletonPoints(create_data, skeleton, include):
    new_points = None
    new_labels = None
    if not create_data == None and len(create_data)>0:
        temp_points = []
        for bone_group, t_value in create_data:
            point = skel.skeletonGetBoneControlPoint(skeleton, bone_group, t_value)
            if len(point)>0:
                temp_points.append(point)

        if len(temp_points)>0:
            new_points = temp_points
            new_labels = [include] * len(new_points)

    return new_points, new_labels

def samProcessBoundingBox(create_data, body_parts):
    new_bbox   = None
    return new_bbox


# Set the expected directory paths ro base images, densepose images and json files
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

# Sets a given image into a SAM prediction model
def setSAMImage(pred, image):
    pred.set_image(image)

# Plots a given pixel mask
def showMask(mask, ax, colour=[0, 0, 255], opacity=0.6):
    print(f"Colour: {colour}, Opacity: {opacity}")
    color = np.array([colour[0]/255, colour[1]/255, colour[2]/255, opacity])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
# Show a series of SAM control points on an image - positive points in green, negative in red
def showPoints(coords, labels, ax, marker_size=375):
    if len(coords)>0:
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# Shows a box, defined as [x0, y0, x1, y1], on an image
def showBox(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Displays a list of boxes
def showBoxes(boxes, ax):
    for box in boxes:
        showBox(box, ax)

# Displays a mask on a base image 
def showImageAndMask(base_image, 
                     mask = None, 
                     title = None, 
                     win_title = None,
                     blackout = False):
        plt.figure(figsize=(10,10))
        if not blackout:
            plt.imshow(base_image)
        else:
            black_image = np.zeros((base_image.shape[0], base_image.shape[1], 3), dtype=np.uint8)
            plt.imshow(black_image)
        
        if not mask==None:
            showMask(mask, plt.gca(), colour=[255,255,0], opacity=0.6)

        if not title==None:
            plt.title(title, fontsize=18)

        plt.axis('on')
        
        if not win_title==None:
            fig_manager = plt.get_current_fig_manager()
            fig_manager.set_window_title(win_title)
        plt.show() 


# Shows mask overlaid on an image plus SAM mode lscoring details
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
input_colours    = ccg.input_mode_dict[input_mode]
grouping_mode    = "RAW" # This sets how densepose 'fine' classifications are aggregated - see 'DenseposeGroupingColours.py'
grouping         = ccg.group_mode_dict[grouping_mode]
image_extensions = ["jpg", "png"]

use_file_dialog  = True

##OK Let's just hardcode these directories for now as those for the 'detectron2' code locations
root_dir = os.getcwd()
parent_of_root = os.path.dirname(root_dir)
in_root_dir = os.path.join(parent_of_root, "detectron2")
base_dir, dens_dir, json_dir, out_dir = setDirectories(in_root_dir, root_dir)

use_file_dialog = False

try:
    # Check to see if directories exist
    if not checkDirectories(base_dir, dens_dir, json_dir, out_dir):
        raise ScriptException("One or more required directories does not exist!")
    
    #OK Get file lists for each directory
    base_list, dens_list, json_list = getDirectoryFileLists(base_dir, dens_dir, json_dir, image_extensions)
 
    #file_idx = 1
    if not use_file_dialog: 
        file_idx = random.randint(0, len(base_list)-1)
        base_file = base_list[file_idx]
        print(f"Random file selected: Base file: {file_idx} : '{base_file}'")
    else:
        base_file = pu.getImageFileByDialog()
        print(f"Files selected: {base_file}")
    
    OK, dens_file, json_file = getFilenameTriple(base_file, dens_list, json_list)
    if not OK:
        raise ScriptException("Matching .json or densepose file to base image failed!")

    # OK We should have the files we need now... load 'em up
    OK, base_image, dens_image, image_metrics, body_parts, skeleton = readFileTriple(base_file, 
                                                                                     dens_file, 
                                                                                     json_file)
    if not OK:
        raise ScriptException("Failed to read in file data!")

    OK, dens_image, image_metrics, body_parts, skeleton = doImageCheckingAndRescaling(base_image, 
                                                                                      dens_image, 
                                                                                      grouping,
                                                                                      image_metrics, 
                                                                                      body_parts, 
                                                                                      skeleton)
    if not OK:
        raise ScriptException("Failed to rescale image or image metrics correctly!")
 
    # OK, Now we should be able to start doing something with SAM
    if is_sam_OK == True:
        print("SAM OK!")
       
        #setSAMImage(base_image)

        #mask = getSAMMask(mask_creation_properties)

    #Try to get torse mask from denspose image
    #print(f"Base shape: {dens_image.shape}")
    #mask, _ = getGroupMask(dens_image, '2-Torso-Front', mask_buffer=1)
    #print(f"Mask shape: {torso_mask.shape}")
    #rep_colour = (255,255,0)
    #mask_colour= (255,0,255)
    #composite = createMaskedImage(base_image, mask, mask_colour, rep_colour=rep_colour)
    #result = createCompositeImage(base_image, composite, mask_colour)
    #showImageAndMask(result)

except ScriptException as e:
    print("EXIT : ", e)

except Exception as e:
    pu.dumpException(e)