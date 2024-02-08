import os
import json
import copy
import math
import random
import skimage as ski
import cv2
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import cvton_colour_groupings as ccg
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import binary_dilation, binary_fill_holes
import pauls_utils as pu
import densepose_functions as df
import skeleton_functions as skel
import sam_pose_templates as spt

##########################################################################################################
## Custom exception class 
class ScriptException(Exception):
    pass

##########################################################################################################

# Adds a buffer around a binary mask area of 'buffer_size' pixels
def addBufferToMask(mask, buffer_size=5):
    buffered_mask = binary_dilation(mask, iterations=buffer_size)
    return buffered_mask

def checkInputRescaling(inputs, template):
    OK = True
    if pu.isValidDict(inputs) and pu.isValidDict(template):
        if pu.doListsMatch(list(inputs.keys()), template['INPUT_KEYS']):
            #print("Passed lists matching..")
            type_list = getInputFileTypes(inputs)
            want_list = ["<class 'numpy.ndarray'>","<class 'numpy.ndarray'>","<class 'dict'>"]
            #print(f"Type list {type_list}")
            #print(f"Want list {want_list}")
            if type_list == want_list:        
                doInputCheckingAndRescaling(inputs, template)
    return OK

def countNonBlackPixels(image):
    grayscale_image = ski.color.rgb2gray(image)
    non_black_pixels = np.count_nonzero(grayscale_image)
    return non_black_pixels

def countTrueValues(array_2d):
    array_np = np.array(array_2d)
    count_true = np.sum(array_np)
    return count_true

def createImageByAddingMask(image, mask, mask_col=[255, 255, 255]):
    label_image = ski.measure.label(mask)
    mask = mask.astype(bool)
    coloured_mask = ski.color.label2rgb(label_image, bg_label=0, colors=[mask_col])
    result = np.copy(image)
    result[mask] = coloured_mask[mask]
    return result

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

def createLowestThresholdMask(mask, T_value, buffer=0):
    if T_value>=0.0 and T_value <= 1.0: 
        head_oobb = getMaskOOBB(mask, flip=True)
        ys = [point[1] for point in head_oobb]
        min_y = min(ys)
        range_y = max(ys) - min_y
        print(f"Mask shape: {mask.shape}, length: {len(mask)}, Type: {type(mask)}")
        print(f"Head_OOBB: {head_oobb}, MinY: {min_y}, Range: {range_y}")
        threshold = round(min_y + (T_value * range_y))
        new_mask = np.zeros_like(mask)
        new_mask[threshold:, :] = mask[threshold:, :]
        if buffer > 0:
            new_mask = addBufferToMask(new_mask, buffer)
        return new_mask

def createMaskedImage(original_image, mask_image, back_colour='black', rep_colour=None):
    back_image = createColouredImage(original_image, colour=back_colour)
    mask_image_coloured = ski.color.gray2rgb(mask_image)
    if not rep_colour==None:    
        original_image = createColouredImage(original_image, colour=rep_colour)    
    composite_image = np.where(mask_image_coloured > 0, original_image, back_image)
    composite_image = ski.img_as_ubyte(composite_image)
    return composite_image

#NB: Processing of the inputs is done by reference, changing the originals 
def doInputCheckingAndRescaling(inputs, template):
    OK = True
    # Get required parameters from the images and json file
    grouping        = ccg.group_mode_dict[inputs['2_JSON']['BODY_PARTS']['GR_MODE']]
    base_image      = inputs['0_BASE']
    dens_image      = inputs['1_ADD_IMAGE']
    base_shape      = base_image.shape
    dens_shape      = dens_image.shape 
    image_metrics   = inputs['2_JSON']['IMAGE_DATA']
    body_metrics    = inputs['2_JSON']['BODY_PARTS']
    skeleton        = inputs['2_JSON']['SKELETON']
    json_shape      = (inputs['2_JSON']['IMAGE_DATA']['IM_SHAPE'][0], inputs['2_JSON']['IMAGE_DATA']['IM_SHAPE'][1], 3)
    json_normalised = inputs['2_JSON']['BODY_PARTS'][next(iter(grouping))]['NORMALISED']
 
    # Does the input image shape match the template output size
    output_shape = pu.getSafeDictKey(template, ['OUTPUT_SIZE'])
    if not output_shape == None:
        i_x = base_image.shape[1]
        i_y = base_image.shape[0]
        o_x = output_shape[0]
        o_y = output_shape[1]
        if not i_x == o_x or not i_y == o_y:
            print("Base image has been resized to match template output!")
            base_image = ski.transform.resize(base_image, (o_y, o_x), anti_aliasing=True)
            base_image = ski.img_as_ubyte(base_image)
            inputs['0_BASE'] = base_image
            base_shape      = base_image.shape

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
            OK = False
        else: 
            print("Images are of correct shape, and metrics do not need to be rescaled")
    else:
        print("Resizing densepose mask image to match base image, and rescaling mask metrics")
        new_image_metrics    = df.getRescaledImageMetrics(image_metrics, base_shape[1], base_shape[0]) 
        new_body_metrics     = df.getRescaledResults(body_metrics, base_shape[1], base_shape[0])
        new_skeleton         = skel.skeletonGetRescaledBones(skeleton, base_shape[1], base_shape[0])
        dens_image           = ski.transform.resize(dens_image, (base_shape[0], base_shape[1]), anti_aliasing=False)
        dens_image           = ski.img_as_ubyte(dens_image)
        
        inputs['1_ADD_IMAGE']          = dens_image
        inputs['2_JSON']['IMAGE_DATA'] = new_image_metrics
        inputs['2_JSON']['BODY_PARTS'] = new_body_metrics
        inputs['2_JSON']['SKELETON']   = new_skeleton        
        
        dens_shape      = inputs['1_ADD_IMAGE'].shape
        json_shape      = (inputs['2_JSON']['IMAGE_DATA']['IM_SHAPE'][0], inputs['2_JSON']['IMAGE_DATA']['IM_SHAPE'][1], 3)
        json_normalised = inputs['2_JSON']['BODY_PARTS'][next(iter(grouping))]['NORMALISED']
   
        print(f"Rescaled Base image shape     : {base_shape}")
        print(f"Rescaled Densepose mask shape : {dens_shape}")
        print(f"Rescaled JSON metrics shape   : {json_shape}")
        print(f"Normalised metrics?           : {json_normalised}")
    
    return OK

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

def fitEllipseToOOBB(coords):
    coords = np.array(coords, dtype=np.float32)
    ellipse = cv2.fitEllipse(coords)
    center, axes, angle = ellipse
    return center, axes, angle

# Creates a mask based on a single, specified colour
def getBinaryMask(image, colour):
    mask = np.all(image == colour, axis=-1)
    return mask, colour

# Creates a mask based on tolerance to a certain colour in linear space
def getBinaryMaskWithTolerance(image, colour, tolerance=20):
    colour_diff = np.linalg.norm(image - colour, axis=-1)
    binary_mask = colour_diff <= tolerance
    return binary_mask, colour

# Gets a composite bounding box based on a number of body parts 
def getCompositeBoundingBox(body_parts, part_list, max_parts=100):
    result = None
    if pu.isValidList(part_list):
        temp_boxes = []
        for body_part in part_list:
            if body_part in body_parts.keys() and max_parts>0:
                bbox_list = body_parts[body_part]['BBOXS']
                if not bbox_list==None and len(bbox_list)>0: 
                    if max_parts<len(bbox_list):
                        temp_boxes.extend(bbox_list)
                    else:
                        temp_boxes.extend(bbox_list[:max_parts])    

        if pu.isValidList(temp_boxes):                
            min_X = min_Y = math.inf
            max_X = max_Y = -math.inf
            for bbox in temp_boxes:
                min_Y = min(min_Y, bbox[0])
                min_X = min(min_X, bbox[1])
                max_Y = max(max_Y, bbox[2])
                max_X = max(max_X, bbox[3])
            
            if min_X < math.inf:
                result = [min_Y, min_X, max_Y, max_X]
    return result

def getEllipseMask(image, center, axes_lengths, angle_degrees):
    mask = np.zeros_like(image)
    mask = ski.color.rgb2gray(mask)
    print(f"getEllipseMask shape: {mask.shape}")

    # Get ellipse coordinates
    rr, cc = ski.draw.ellipse(center[0], center[1], axes_lengths[0] // 2, axes_lengths[1] // 2, rotation=np.radians(angle_degrees))

    # Set ellipse pixels to 1
    mask[rr, cc] = 1

    # Fill holes in the mask
    filled_mask = binary_fill_holes(mask)

    return filled_mask

# Create a composite mask based on lookup of body parts by densepose grouping dictionary key
def getGroupMask(image, 
                 group_name, 
                 group_mode='RAW', 
                 input_mode='CVTON', 
                 mask_buffer=0, 
                 verbose = False):
    
    grouping         = ccg.group_mode_dict[group_mode]
    input_colours    = ccg.input_mode_dict[input_mode]
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

# Get a list of the read in data types from an input list
def getInputFileTypes(inputs):
    type_list = []
    if pu.isValidDict(inputs):
        for _, item in inputs.items():
            type_list.append(str(type(item)))
    return type_list  

# Gets an OOBB for a given mask image using cv2
def getMaskOOBB(mask, flip=False):
    points = np.array(maskToPoints(mask), dtype=np.float32)
    rect = cv2.minAreaRect(points)
    box_vertices = cv2.boxPoints(rect).astype(int)
    
    if flip==True:
        box_vertices = [[x, y] for y, x in box_vertices]
    else:
        box_vertices = [[x, y] for x, y in box_vertices]
    return box_vertices

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

# Convert an image mask to a numpy array of white pixels
def maskToPoints(mask):
    # Find white pixels in the mask
    white_pixels = np.column_stack(np.where(mask > 0))
    return white_pixels

def moveOOBB(obb_coords, displacement):
    # Ensure that there are exactly four coordinates
    if len(obb_coords) != 4:
        raise ValueError("Four coordinates are required to define an oriented bounding box.")

    # Convert coordinates to NumPy array
    obb_coords = np.array(obb_coords)

    # Calculate the centroid of the OBB
    centroid = np.mean(obb_coords, axis=0)

    # Determine the major axis (assumed to be the line connecting the first and second vertices)
      
    major_axis = obb_coords[1] - obb_coords[0]
    major_axis /= np.linalg.norm(major_axis)

    # Calculate the displacement vector along the major axis
    displacement_vector = displacement * major_axis

    # Apply the displacement to each vertex
    moved_obb_coords = obb_coords + displacement_vector

    return moved_obb_coords

# Read in all required fiels
def readAssociatedFiles(base_file, additional_files):
    files = {}
    exts = []
    try:
        print(f"Reading base image file: '{base_file}'")
        files['0_BASE'] = ski.io.imread(base_file)
        ext = os.path.splitext(base_file)[1]
        exts.append(ext)
        
        if pu.isValidList(additional_files):
            for i, add_file in enumerate(additional_files):
                # Get the extension of the file
                ext = os.path.splitext(add_file)[1]
                #print (f"File load on extension {ext}")
                if (ext == '.json'):
                    print(f"Reading image json file: '{add_file}'")
                    with open(add_file, 'r') as file:
                        key = str(i+1) + '_JSON' 
                        files[key] = json.load(file)
                        exts.append(ext)                    
                else:
                    print(f"Reading add image file : '{add_file}'")
                    key = str(i+1) + '_ADD_IMAGE'
                    files[key] = ski.io.imread(add_file)
                    exts.append(ext)       
            
    except Exception as e:
        print(f"ERROR!: Failed to read inputs associated with '{base_file}'")
        pu.dumpException(e)    
        pass

    return files, exts

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

def samCreateCompatibleInputs(create_props, 
                              inputs,
                              verbose = False):
    
    points      = []
    labels      = []
    bbox        = None
    mask        = None
    multi_out   = True
    force_multi = False

    body_parts    = pu.getSafeDictKey(inputs, ['2_JSON','BODY_PARTS'])
    skeleton      = pu.getSafeDictKey(inputs, ['2_JSON','SKELETON'])
    image_metrics = pu.getSafeDictKey(inputs, ['2_JSON','IMAGE_DATA'])

    #check dictionary keys
    if not create_props['AI_POINTS'] == None:
        print("Adding AI points")
        new_points, new_labels = samProcessAbsolutePoints(create_props['AI_POINTS'], True)
        if pu.isValidList(new_points) and pu.isValidList(new_labels):
            points.extend(new_points)
            labels.extend(new_labels)

    if not create_props['RI_POINTS'] == None and not image_metrics == None:
        print("Adding RI points")
        new_points, new_labels = samProcessRelativePoints(create_props['RI_POINTS'], image_metrics, True)
        if pu.isValidList(new_points) and pu.isValidList(new_labels):
            points.extend(new_points)
            labels.extend(new_labels)

    if not create_props['AE_POINTS'] == None:
        print("Adding AE points")
        new_points, new_labels = samProcessAbsolutePoints(create_props['AE_POINTS'], False)
        if pu.isValidList(new_points) and pu.isValidList(new_labels):
            points.extend(new_points)
            labels.extend(new_labels)

    if not create_props['RE_POINTS'] == None and not image_metrics == None:
        print("Adding RE points")
        new_points, new_labels = samProcessRelativePoints(create_props['RE_POINTS'], image_metrics, False)
        if pu.isValidList(new_points) and pu.isValidList(new_labels):
            points.extend(new_points)
            labels.extend(new_labels)

    if not create_props['SI_POINTS'] == None and not skeleton == None:
        print("Adding SI points")
        new_points, new_labels = samProcessSkeletonPoints(create_props['SI_POINTS'], skeleton, True)
        if pu.isValidList(new_points) and pu.isValidList(new_labels):
            points.extend(new_points)
            labels.extend(new_labels)

    if not create_props['SE_POINTS'] == None and not skeleton == None:
        print("Adding SE points")
        new_points, new_labels = samProcessSkeletonPoints(create_props['SE_POINTS'], skeleton, False)
        points.extend(new_points)
        labels.extend(new_labels)

    if not create_props['BBOX'] == None and not image_metrics == None and not body_parts == None:
        print("Adding Bounding Box")
        bbox = samProcessBoundingBox(create_props['BBOX'], body_parts, [0, 0, image_metrics['IM_SHAPE'][1], image_metrics['IM_SHAPE'][0]])
        
    if not create_props['MASK'] == None:
        print("Adding Mask")
        mask = create_props['MASK']
    
    multi_out = create_props['MULTI_OUT'] 
    print(f"Set multimask output to {multi_out}")

    if not create_props['FORCE_MULTI'] == None:
        if create_props['FORCE_MULTI']>=0 and create_props['FORCE_MULTI']<=2:
            force_multi = create_props['FORCE_MULTI']
            print(f"Set forced multimask return to {force_multi}")
        
    if len(points)>0 and len(labels)>0 and len(points)==len(labels):
        points = np.array(points)
        labels = np.array(labels)
    else:
        points = None
        labels = None

    if not bbox==None:
        bbox = np.array(bbox)

    if verbose==True:
        print("----------------------------------------------")
        print("Created SAM inputs:")
        print(f"Points : {points}")
        print(f"Labels : {labels}")   
        print(f"B-Box  : {bbox}")   
        print(f"Mask   : {mask}")   
        print(f"Multi  : {multi_out}")   
        print(f"Forced : {force_multi}")           
        print("----------------------------------------------")

    return points, labels, bbox, mask, multi_out, force_multi

def samDoInference(layer,
                   sam_predictor,
                   sam_setup,
                   inputs,
                   verbose=False, 
                   show_intermediate_results=False, 
                   show_final_result=False):
    result = None

    print(f"{inputs.keys()}")
    try:
        base_image                = pu.getSafeDictKey(inputs, ['0_BASE'])
        image_metrics             = pu.getSafeDictKey(inputs, ['2_JSON','IMAGE_DATA'])
        if not pu.isValidNpArray(base_image) or not pu.isValidDict(image_metrics):
            raise ScriptException("WARNING! : SOmething went wrong retrieving base image or image metrics from input dictionary!")
        
        samSetImage(sam_predictor, base_image)
        points, labels, bbox, mask, multi_out, force_multi = samCreateCompatibleInputs(sam_setup, 
                                                                                       inputs, 
                                                                                       verbose = verbose)
        
        masks =[]
        scores=[]
        logits=[]

        if pu.isValidNpArray(points):    
            masks, scores, logits = sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                mask_input=mask,
                box=bbox,
                multimask_output=multi_out
            )
            
        if len(masks)>0:
            if multi_out==True: #Return highest score mask
                if not force_multi:
                    idx = np.argmax(scores)
                    print(f"Selected mask idx: {idx}")
                    result = masks[np.argmax(scores)]
                else:
                    result = masks[force_multi]
            else:
                result = masks[0]

            # Mask image post-processing goes here
            if not sam_setup['ADD_BUFFER']==None and sam_setup['ADD_BUFFER']>0:
                result = addBufferToMask(result, 
                                        sam_setup['ADD_BUFFER'])

            if sam_setup['FILL_HOLES']==True:
                result = binary_fill_holes(result, 
                                        structure=disk(10))

            if show_intermediate_results==True:
                showImageAndMaskDetail(layer,
                                       base_image, 
                                       masks, 
                                       scores, 
                                       points, 
                                       bbox, 
                                       labels, 
                                       image_metrics['FILE'])                 

            if show_final_result==True:
                showImageAndMask(base_image,
                                 base_image,
                                 result,
                                 title = f"Final result mask: {layer}",
                                 win_title = image_metrics['FILE'])
    except ScriptException as e:
        pu.dumpException(e)

    return result

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
            new_points = [[round(x * w), round(y * h)] for x, y in create_data]
            new_labels = [include] * len(new_points)
    return new_points, new_labels

def samProcessSkeletonPoints(create_data, skeleton, include):
    new_points = None
    new_labels = None
    if not create_data == None and len(create_data)>0:
        temp_points = []
        for bone_group, t_value in create_data:
            print(f"Bone group: {bone_group}, : t_value: {t_value}")
            #print(f"{skeleton}")
            point = skel.skeletonGetBoneControlPoint(skeleton, bone_group, t_value)
            print(f"Bone: {bone_group}, T_Val: {t_value}, Point: {point}")
            if len(point)>0:
                if point[2]==True:
                    temp_points.append([point[1], point[0]])

        if len(temp_points)>0:
            new_points = temp_points
            new_labels = [include] * len(new_points)

    return new_points, new_labels

def samProcessBoundingBox(create_data, body_parts, clamp_vals=None):
    new_bbox   = None
    if not create_data == None and len(create_data)>0:
        temp_bbox = getCompositeBoundingBox(body_parts, create_data[0])
        if not temp_bbox==None:
            new_bbox = []
            ox = round((temp_bbox[1]+temp_bbox[3])*0.5)
            oy = round((temp_bbox[0]+temp_bbox[2])*0.5)
            w  = temp_bbox[3]-temp_bbox[1]
            h  = temp_bbox[2]-temp_bbox[0]
            #print(f"Box origin : [{ox}, {oy}], width: {w}, height: {h} ")
            #print(f"Temp box   : {temp_bbox}")
            if not create_data[1]==None and len(create_data[1])==2:
                sw = w * create_data[1][0] * 0.5
                sh = h * create_data[1][1] * 0.5
                temp_bbox[0] = ox - sw
                temp_bbox[1] = oy - sh
                temp_bbox[2] = ox + sw
                temp_bbox[3] = oy + sh
                #print(f"Scaling    : {create_data[1]}")
                #print(f"Temp box(s): {temp_bbox}")
            if not create_data[2]==None and len(create_data[2])==4:
                temp_bbox[0] = temp_bbox[0] - create_data[2][0]
                temp_bbox[1] = temp_bbox[1] - create_data[2][1]
                temp_bbox[2] = temp_bbox[2] + create_data[2][2]
                temp_bbox[3] = temp_bbox[3] + create_data[2][3]
                #print(f"Padding    : {create_data[2]}")
                #print(f"Temp box(p): {temp_bbox}")
            if not clamp_vals==None and len(clamp_vals)==4:
                temp_bbox[0] = max(temp_bbox[0], clamp_vals[0])
                temp_bbox[1] = max(temp_bbox[1], clamp_vals[1])
                temp_bbox[2] = min(temp_bbox[2], clamp_vals[2])
                temp_bbox[3] = min(temp_bbox[3], clamp_vals[3])
                #print(f"Clamped    : {clamp_vals}")
                #print(f"Temp box(c): {temp_bbox}")                    

                new_bbox.append(temp_bbox)
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
def samSetImage(pred, image):
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
    if pu.isValidNpArray(coords):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        #print(f"Plotting points : {coords}")
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# Shows a box, defined as [x0, y0, x1, y1], on an image
def showBox(box, ax):
    if pu.isValidNpArray(box):
        #print(f"Box: {box}")
        x0 = min(box[0][0], box[0][2])
        y0 = min(box[0][1], box[0][3])
        w, h = round(box[0][2] - box[0][0]), round(box[0][3] - box[0][1])
        #print(f"showBox: x0: {x0}, y: {y0}, w: {w}, h: {h}")
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Displays a list of boxes
def showBoxes(boxes, ax):
    if pu.isValidNpArray(boxes):
        for box in boxes:
            showBox(box, ax)

# Displays a mask on a base image 
def showImageAndMask(base_image, 
                     second_image, 
                     mask = None, 
                     title = None, 
                     win_title = None,
                     blackout = False):
        plt.figure(figsize=(10,10))
        plt.subplot(1, 2, 1)
        plt.imshow(base_image)
        plt.title('Original Image', fontsize=18)       
        
        plt.subplot(1, 2, 2)
        if not blackout:
            plt.imshow(second_image)
        else:
            black_image = np.zeros((base_image.shape[0], base_image.shape[1], 3), dtype=np.uint8)
            plt.imshow(black_image)
        plt.title('Image + Mask', fontsize=18)

        if pu.isValidNpArray(mask):
            showMask(mask, plt.gca(), colour=[255,255,0], opacity=0.6)

        #if not title==None:
        #    plt.title(title, fontsize=18)
        #
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
        print(f"Mask: {i}, Score: {score}")
        plt.figure(figsize=(10,10))
        if not blackout:
            plt.imshow(base_image)
        else:
            mask_colour=[255,255,255]
            black_image = np.zeros((base_image.shape[0], base_image.shape[1], 3), dtype=np.uint8)
            plt.imshow(black_image)
        showMask(mask, plt.gca(), colour=mask_colour, opacity=1.0)
        showPoints(points, labels, plt.gca())
        showBox(bboxes, plt.gca())
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

##OK Let's just hardcode these directories for now as those for the 'detectron2' code locations
root_dir        = os.getcwd()
parent_of_root  = os.path.dirname(root_dir)
in_root_dir     = parent_of_root
out_root_dir    = root_dir
do_saves        = True
do_input_saves  = False

use_random_file = True
use_file_dialog = False

verbose                   = False
show_intermediate_results = False
show_final_result         = False
show_final_composite      = True

try:
    # Set up the SAM template being used...
    #sam_template = spt.samBodyTemplateWithHands()
    #sam_template = spt.samTemplateImageParse()
    #sam_template = spt.samTemplateGarment()
    #sam_template = spt.samTemplateGarmentMask()
    #sam_template = spt.samTemplateGarmentAgnostic32()
    #sam_template = spt.samTemplateSemanticBodyLabels()
    sam_template = spt.samHairTemplate()
   
    if not pu.isValidDict(sam_template):
        raise ScriptException("ERROR! : Could not retrieve a valid template dictionary!")        

    # Create the base file list, either by dialog, or by reading the directory
    if use_file_dialog==True:
        base_file = pu.getImageFileByDialog()
        base_file_list = [base_file]
        print(f"File selected: {base_file}")
    else:
        # Check that the base directory exists
        add_in_dir = sam_template['BASE_IMAGE_INPUT_PATH']
        base_dir = os.path.join(in_root_dir, add_in_dir)
        base_file_format = sam_template['BASE_IMAGE_INPUT_FORMATS']
        if not pu.checkDirectory(base_dir):
            raise ScriptException(f"ERROR! : Input path '{base_dir}' does not exist!")
        else: # Get the base file list
            print(f"Base input directory set to: '{base_dir}'.")
            base_file_list = pu.getFilesInDirectory(base_dir, base_file_format)
            if use_random_file==True: # Pick a single file from the list
               file_idx = random.randint(0, len(base_file_list)-1)
               base_file = base_file_list[file_idx]
               base_file_list = [base_file] 

    # Check to see if any further input directories exist and, if so, generate file lists for these 
    # that match the base file list
    extra_dirs   = sam_template['ADDITIONAL_INPUT_PATHS']
    file_formats = sam_template['ADDITIONAL_INPUT_FORMATS'] 
    extra_dir_files = {}
    output_paths    = {}
    if pu.isValidList(extra_dirs) and pu.isValidList(file_formats) and len(extra_dirs) == len(file_formats):
        for dir, file_format in zip(extra_dirs, file_formats):
            #print(f"In root '{in_root_dir}'...")
            full_ext_dir = os.path.join(in_root_dir, dir)
            #print(f"Checking '{full_ext_dir}'...")
            if not pu.checkDirectory(full_ext_dir):
                raise ScriptException(f"ERROR! : Additional required path '{dir}' does not exist!")
            else:
                print(f"Checking files in '{full_ext_dir}'")
                match_file_list = []
                temp_file_list = pu.getFilesInDirectory(full_ext_dir, file_format)
                for base_file in base_file_list:
                    match_file = pu.getMatchingFileList(base_file, temp_file_list)
                    match_file_list.append(match_file)
                extra_dir_files[dir] = match_file_list
                #print(f"{extra_dir_files}")
                if not pu.isValidDict(extra_dir_files):
                    raise ScriptException(f"ERROR! Failed to build file dictionary for : '{dir}'!")
   
    # Loop over all files in the base_file_list and try to proccess their masks
    print(f"Processing a total of {len(base_file_list)} file(s).")
    for file_idx, base_file in enumerate(base_file_list):
        print("--------------------------------------")
        print(f"PROCESSING: Base file: '{base_file}'.")

        additional_files = []
        for j, extra_dir in enumerate(extra_dir_files):
            add_file_list = extra_dir_files[extra_dir]
            add_file = add_file_list[file_idx]
            #print(f"Add file : '{add_file}'")
            additional_files.append(add_file)
        
        inputs, actual_exts = readAssociatedFiles(base_file, additional_files)   
        #print(f"Input keys        : {list(inputs.keys())}")
        #print(f"Actual input exts : {actual_exts}")

        if not pu.isValidDict(inputs):
            raise ScriptException("ERROR! : Failed to read input data!")

        if not checkInputRescaling(inputs, sam_template)==True:
            raise ScriptException("ERROR! : Failed to rescale image or image metrics correctly!")
    
        # Check and create output directories 
        add_out_dirs = sam_template['OUTPUT_PATHS']
        out_key_list = ['MASK']
        out_key_list.extend(list(inputs.keys())) 
        print(f"out_key_list : {out_key_list}.")
        #print(f"Output path length : {len(add_out_dirs)}, Input path length: {len(inputs)+1}.")
        
        if pu.isValidList(add_out_dirs):
            do_saves = True
            output_formats = ['png']
            if len(add_out_dirs)==(len(inputs)+1):
                output_formats.extend(actual_exts)         
                print(f"Aassuming we're saving inputs as outputs!")
                #print(f"Output formats : {output_formats}.")
                
            for i, path in enumerate(add_out_dirs):
                full_output_path = os.path.join(out_root_dir, path)
                if not pu.checkAndCreateDirectory(full_output_path):
                    raise ScriptException(f"ERROR! : Failed to find and/or create output directory '{full_output_path}!'!")
                else:
                    output_paths[out_key_list[i]] = [full_output_path, output_formats[i]]
            #print(f"Output dictionary : {output_paths}")             
        else:
            print("WARNING! : Could not get output directories from template. Outputs will not be saved!")
            do_saves = False
        
        # OK, Now we should be able to start doing something with SAM
        if is_sam_OK == True:
            print("SAM OK!")
            stored_masks              = {}

            print("Creating output image template.")                
            if sam_template['BACKGROUND_TYPE'] == 'SOLID_FILL':
                output_image = createColouredImage(inputs['0_BASE'], colour=sam_template['BACKGROUND_COLOUR'])
            elif sam_template['BACKGROUND_TYPE'] == '0_BASE':
                output_image = inputs['0_BASE']
            else:
                raise ScriptException("ERROR! : Unknown background type for output image!")
            
            # Process each mask key in turn
            process_order = sam_template['PROCESSING_ORDER']
            if pu.isValidList(process_order):
                for layer in process_order:
                    print(f"Processing mask layer: {layer}")
                    sam_setup = sam_template[layer]
                    mask_to_add = samDoInference(layer,
                                                 predictor,
                                                 sam_setup,
                                                 inputs, 
                                                 verbose=verbose,
                                                 show_intermediate_results=show_intermediate_results,
                                                 show_final_result = show_final_result)
                                        
                    if pu.isValidNpArray(mask_to_add):
                        if sam_setup['MODE'] == "ADDITIVE":
                            print("Adding part mask to output mask")
                            output_image = createImageByAddingMask(output_image, 
                                                                   mask_to_add,
                                                                   mask_col = sam_setup['OUTPUT_COLOUR'])
                        elif sam_setup['MODE'] == "BASE_MASK":
                            print("Adding part mask to output mask")
                            output_image = createMaskedImage(inputs['0_BASE'], 
                                                             mask_to_add,
                                                             back_colour = sam_template['BACKGROUND_COLOUR'])
                        else: 
                            raise ScriptException(f"ERROR! : Unknown mode {sam_setup['MODE']}") 

                        if sam_setup['STORE_MASK'] == True:
                            stored_masks[layer] = copy.copy(mask_to_add)
                            print("Storing part mask for further processing")
                    print("----------------------")                    

                #print(f"{sam_template.keys()}")
                print(f"Stored masks dictionary valid? : {pu.isValidDict(stored_masks)}")
                print(f"Stored mask dictionary keys    : {list(stored_masks.keys())}")
                if "POST_PROCESSING" in sam_template.keys() and pu.isValidDict(stored_masks):
                    print("Doing post-processing...")
                    #print(f"{sam_template['POST_PROCESSING']}")
                    if "ADD_NECK" in sam_template['POST_PROCESSING']:
                        print("Applying 'ADD_NECK' post_process operation.")
                        if "HEAD" in stored_masks.keys():
                            neck_colour = pu.getSafeDictKey(sam_template, ['POST_PROCESSING', 'ADD_NECK', 'OUT_COLOUR'])
                            threshold = pu.getSafeDictKey(sam_template, ['POST_PROCESSING', 'ADD_NECK', 'THRESHOLD'])
                            neck_mask = createLowestThresholdMask(stored_masks['HEAD'], T_value = threshold)
                                                        
                            #print(f"Neck colour : {neck_colour}.")
                            output_image = createImageByAddingMask(output_image, 
                                                                   neck_mask, 
                                                                   mask_col=neck_colour)
                        else:
                            print(f"WARNING ! : Couldn't find stored mask 'HEAD' in '{stored_masks}'!")
                
                    if "REMOVE_BODY" in sam_template['POST_PROCESSING']:
                        print("Applying 'REMOVE_BODY' post_process operation.")
                        if "BODY" in stored_masks.keys():
                            body_colour = pu.getSafeDictKey(sam_template, ['POST_PROCESSING', 'REMOVE_BODY', 'OUT_COLOUR'])
                            threshold = pu.getSafeDictKey(sam_template, ['POST_PROCESSING', 'REMOVE_BODY', 'THRESHOLD'])
                            buffer = pu.getSafeDictKey(sam_template, ['POST_PROCESSING', 'REMOVE_BODY', 'BUFFER'])
                            body_mask = createLowestThresholdMask(stored_masks['BODY'], T_value = threshold, buffer=buffer)
                                                            
                            #print(f"Neck colour : {neck_colour}.")
                            output_image = createImageByAddingMask(output_image, 
                                                                   body_mask, 
                                                                   mask_col=body_colour)
                        else:
                            print(f"WARNING ! : Couldn't find stored mask 'BODY' in '{stored_masks}'!")
                    
                else: 
                    print("No post-processing defined.")
                
                print("----------------------")
            if show_final_composite==True:   
                showImageAndMask(inputs['0_BASE'], output_image, win_title=base_file)
                        
            if do_saves==True:
                file_title = pu.getFileTitle(base_file)
                rename_files = pu.getSafeDictKey(sam_template, ['RENAME_OUTPUT_FILES'])
                print(f"Rename files : {rename_files}")
                if not rename_files == None:
                    if rename_files==True:
                        file_title = str(file_idx+1).zfill(6) + "_0"                
                
                for dir_key, dir_item in output_paths.items():
                    #print(f"{dir_key}, {dir_item[0]}, {dir_item[1]}")
                    output_filename = dir_item[0] + '/' + file_title + "." + dir_item[1]
                    if dir_key == "MASK": 
                        out_data = output_image    
                    else:
                        #print(f"Saving output for key '{dir_key}'")
                        out_data = inputs[dir_key]                    

                    out_type = str(type(out_data))

                    if out_type == "<class 'numpy.ndarray'>":       
                        try:
                            ski.io.imsave(output_filename, out_data)
                            print(f"Saved '{output_filename}'.")
                        except Exception as e:
                            print(f"WARNING! : Failed to save image: {e}")
                    elif out_type == "<class 'dict'>": 
                        try:
                            with open(output_filename, 'w') as json_file:
                                json.dump(out_data, json_file, indent=4)
                                print(f"Saved '{output_filename}'.")
                        except Exception as e:
                            print(f"WARNING! : Failed to save JSON file: {e}")
                    else:
                        print(f"WARNING! : Output data was 'None'!")
            
            print("----= Finished image =----")
            
    print("---------= End of Script =---------")

except ScriptException as e:
    print("EXIT : ", e)

except Exception as e:
    pu.dumpException(e)