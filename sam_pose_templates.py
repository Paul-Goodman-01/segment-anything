import cvton_colour_groupings as ccg

sam_templates_dict = {"BODY_PARSE"             : "samTemplateImageBodyParse",
                      "BODY_PARSE_ALT"         : "samTemplateImageBodyParseAlternate",
                      "IMAGE_PARSE_WITH_HANDS" : "samTemplateImageParseWithHands",
                      "IMAGE_PARSE"            : "samTemplateImageParse",
                      "GARMENT"                : "samTemplateGarment",
                      "GARMENT_MASK"           : "samTemplateGarmentMask",
                      "AGNOSTIC_3.2"           : "samTemplateGarmentAgnostic32"}

# Get a default dictionary for a body part
def getDefaultSAMInputDictionary():
    result = {}
    result['IMAGE'] = '0_BASE'
    result['MODE'] = 'ADDITIVE'
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
    result['ADD_BUFFER'] = None
    result['FILL_HOLES'] = False
    result['OUTPUT_COLOUR'] = [0, 0, 0]
    result['STORE_MASK'] = False
    return result

# A template to extract hair from an image
# Intended as a pre-processing step for inclusion in other
# mask generation processes. 
def samHairTemplate():
    template = {}
    template['PRE_PROCESSOR'] = True

    # What directories are required
    template['BASE_IMAGE_INPUT_PATH'] = "detectron2/data"
    template['BASE_IMAGE_INPUT_FORMATS'] = ["jpg","png"]

    template['ADDITIONAL_INPUT_PATHS'] = ["detectron2/results_cvton", "detectron2/results_cvton_dump"]
    template['ADDITIONAL_INPUT_FORMATS'] = [["png"],["json"]]
    template['INPUT_KEYS'] = ['0_BASE','1_ADD_IMAGE','2_JSON']
    template['INPUT_COLOURS'] = 'CVTON'
   
    template['OUTPUT_SIZE']   = [384, 512]
    template['OUTPUT_PATHS'] = []
    template['RENAME_OUTPUT_FILES'] = False

    # Setup Background
    template['BACKGROUND_TYPE'] = "SOLID_FILL"
    template['BACKGROUND_COLOUR'] = [0, 0, 0]
   
    # Overall body definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 2
    sam_setup['OUTPUT_COLOUR'] = [0, 128, 0]
    sam_setup["STORE_MASK"] = True
    template['BODY'] = sam_setup

    # Torso definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = [128, 0, 128]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['TOP_GARMENT'] = sam_setup

    # Head definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup['SI_POINTS'] = [["HEAD", 0.1]]
    sam_setup["SE_POINTS"] = [["ARLO", 0.5],["ALLO",0.5],["ARLI",0.5],["ARLO", 0.5]]
    sam_setup["BBOX"] = [["23-Head-Right","24-Head-Left"], [1.05, 1.7], None]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = [128, 0, 192]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['HEAD'] = sam_setup
                
    # Mask building order
    template['PROCESSING_ORDER'] = ["BODY", 
                                    "HEAD",
                                    "TOP_GARMENT"]

    # Post-processing actions 
    template["POST_PROCESSING"] = {"REMOVE_BODY" : {"OUT_COLOUR" : [0, 0, 0],
                                                    "THRESHOLD"  : 0.40,
                                                    "BUFFER"     : 10},
                                   "GET_MASK"    : {'COLOUR'     : [0,128,0],
                                                    "BUFFER"     : 0},
                                   "SAVE_MASK"   : {"NAME"       : "HAIR"}}
    template['POST_PROCESSING_ORDER'] = ["REMOVE_BODY", "GET_MASK", "SAVE_MASK"]

    return template        

# Template for C-VTON compatible 'body parse' inputs
# Relies on the hair template pre-processor being available
# This version uses the base image 
def samTemplateImageBodyParse():
    template = {}
    template['PRE_PROCESSOR'] = False
    template['USE_PRE_PROCESSOR_LIST'] = ['samHairTemplate']

    # What directories are required
    template['BASE_IMAGE_INPUT_PATH'] = "detectron2/data"
    template['BASE_IMAGE_INPUT_FORMATS'] = ["jpg","png"]

    template['ADDITIONAL_INPUT_PATHS'] = ["detectron2/results_cvton", "detectron2/results_cvton_dump"]
    template['ADDITIONAL_INPUT_FORMATS'] = [["png"],["json"]]
    template['INPUT_KEYS'] = ['0_BASE','1_ADD_IMAGE','2_JSON']
    template['INPUT_COLOURS'] = 'CVTON'
   
    template['OUTPUT_SIZE']   = [384, 512]
    template['OUTPUT_PATHS'] = ["data/viton/data/image_body_parse"]
    template['RENAME_OUTPUT_FILES'] = True

    # Setup Background
    template['BACKGROUND_TYPE'] = "SOLID_FILL"
    template['BACKGROUND_COLOUR'] = ccg.semantic_body_dict["6-Background"]
   
    # Overall body definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 2
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["13-Torso"]
    template['BODY'] = sam_setup

    # Head definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup['SI_POINTS'] = [["HEAD", 0.1]]
    sam_setup["SE_POINTS"] = [["ARLO", 0.5],["ALLO",0.5],["ARLI",0.5],["ARLO", 0.5]]
    sam_setup["BBOX"] = [["23-Head-Right","24-Head-Left"], [1.05, 1.7], None]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["0-Head"]
   
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    sam_setup["STORE_MASK"] = True
    template['HEAD'] = sam_setup

    # Left leg definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["LLUF",0.25],["LLUF",0.5]]
    sam_setup["BBOX"] = [["10-Leg-Left-Upper-Front"], [1.2, 1.2], None]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 1
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["5-Left-Leg"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['LEFT_LEG'] = sam_setup

    # Right leg definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["LRUF",0.25],["LRUF",0.5]]
    sam_setup["BBOX"] = [["9-Leg-Right-Upper-Front"], [1.2, 1.2], None]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 1
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["8-Right-Leg"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['RIGHT_LEG'] = sam_setup

    # Left Hand definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["HLLI", 0.5],["HLLO",0.5]]
    sam_setup["BBOX"] = [["4-Hand-Left"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["1-Left-Hand"]
    sam_setup["ADD_BUFFER"] = 4
    sam_setup["FILL_HOLES"] = True
    template['LEFT_HAND'] = sam_setup

    # Right Hand definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["HRLI", 0.5],["HRLO",0.5]]
    sam_setup["BBOX"] = [["3-Hand-Right"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["3-Right-Hand"]
    sam_setup["ADD_BUFFER"] = 4
    sam_setup["FILL_HOLES"] = True
    template['RIGHT_HAND'] = sam_setup
                
    # Left upper arm definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["ALUO", 0.75], ["ALUI", 0.75]]
    #sam_setup["SE_POINTS"] = [["ARLO", 0.90],["ARLO",0.5],["ARLI",0.5],["LLUF",0.1],["HLLI",0.5],["HLLO",0.5],["HEAD",0.5]]
    sam_setup["BBOX"] = [["15-Arm-Left-Upper-Inner", "17-Arm-Left-Upper-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["2-Left-Arm-Upper"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['LEFT_ARM_UPPER'] = sam_setup         

    # Left lower arm definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["ALLO", 0.75],["ALLI", 0.75]]
    #sam_setup["SE_POINTS"] = [["ARLO", 0.90],["ARUO",0.5],["ARUI",0.5],["LRLUF",0.1],["HLLI",0.5],["HLLO",0.5],["HEAD",0.5]]
    sam_setup["BBOX"] = [["19-Arm-Left-Lower-Inner", "21-Arm-Left-Lower-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["14-Left-Arm-Lower"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['LEFT_ARM_LOWER'] = sam_setup        

    # Right upper arm definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["ARUO", 0.75],["ARUI", 0.75]]
    #sam_setup["SE_POINTS"] = [["ARLO", 0.90],["ARLO",0.5],["ARLI",0.5],["LLUF",0.1],["HLLI",0.5],["HLLO",0.5],["HEAD",0.5]]
    sam_setup["BBOX"] = [["16-Arm-Right-Upper-Inner","18-Arm-Right-Upper-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["7-Right-Arm-Upper"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['RIGHT_ARM_UPPER'] = sam_setup         

    # Right lower arm definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["ARLO", 0.75],["ARLI", 0.75]]
    #sam_setup["SE_POINTS"] = [["ARLO", 0.90],["ARUO",0.5],["ARUI",0.5],["LRLUF",0.1],["HLLI",0.5],["HLLO",0.5],["HEAD",0.5]]
    sam_setup["BBOX"] = [["20-Arm-Right-Lower-Inner", "22-Arm-Right-Lower-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_body_dict["11-Right-Arm-Lower"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['RIGHT_ARM_LOWER'] = sam_setup       

    #Mask building order
    template['PROCESSING_ORDER'] = ["BODY", 
                                    "LEFT_LEG",
                                    "RIGHT_LEG",
                                    "LEFT_ARM_LOWER",
                                    "RIGHT_ARM_LOWER",
                                    "LEFT_ARM_UPPER",
                                    "RIGHT_ARM_UPPER",
                                    "LEFT_HAND",
                                    "RIGHT_HAND",
                                    "HEAD"]
    
    # Post-processing options - add the neck to torso and add hair to head
    # NB: Not tested, but reversing the order of these two may yield better results?
    template["POST_PROCESSING"] = {"ADD_NECK"  : {"OUT_COLOUR" : ccg.semantic_body_dict["13-Torso"],
                                                  "THRESHOLD" : 0.85},
                                   "ADD_SAVED" : {"NAME" : "HAIR",
                                                  "OUT_COLOUR" : ccg.semantic_body_dict["0-Head"]}}
    template["POST_PROCESSING_ORDER"] = ["ADD_NECK", "ADD_SAVED"]

    return template        

# Alternate Template for C-VTON compatible 'body parse' inputs
# Relies on the hair template pre-processor being available
# This version uses both the base and densepose images
def samTemplateImageBodyParseAlternate():
    template = samTemplateImageBodyParse()
    template['BODY']["IMAGE"] = "1_ADD_IMAGE" 
    template['LEFT_LEG']["IMAGE"] = "1_ADD_IMAGE"
    template['RIGHT_LEG']["IMAGE"] = "1_ADD_IMAGE"
    template['LEFT_ARM_LOWER']["IMAGE"] = "1_ADD_IMAGE"
    template['RIGHT_ARM_LOWER']["IMAGE"] = "1_ADD_IMAGE"
    template['LEFT_HAND']["IMAGE"] = "1_ADD_IMAGE"
    template['RIGHT_HAND']["IMAGE"] = "1_ADD_IMAGE"    
    template['HEAD']["IMAGE"] = "1_ADD_IMAGE"
    return template 

# A template for C-VTON 'image parse with hands' masks
def samTemplateImageParseWithHands():
    template = {}
    template['PRE_PROCESSOR'] = False
    
    # What directories are required
    template['BASE_IMAGE_INPUT_PATH'] = "detectron2/data"
    template['BASE_IMAGE_INPUT_FORMATS'] = ["jpg","png"]

    template['ADDITIONAL_INPUT_PATHS'] = ["detectron2/results_cvton", "detectron2/results_cvton_dump"]
    template['ADDITIONAL_INPUT_FORMATS'] = [["png"],["json"]]
    template['INPUT_KEYS'] = ['0_BASE','1_ADD_IMAGE','2_JSON']
    template['INPUT_COLOURS'] = 'CVTON'
   
    template['OUTPUT_SIZE']   = [384, 512]
    template['OUTPUT_PATHS'] = ["data/viton/data/image_parse_with_hands",
                                "data/viton/data/images",
                                "data/viton/data/image_densepose_parse",
                                "data/viton/data/pose"]
    template['RENAME_OUTPUT_FILES'] = True

    # Setup Background
    template['BACKGROUND_TYPE'] = "SOLID_FILL"
    template['BACKGROUND_COLOUR'] = [0, 0, 0]
   
    # Overall body definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 2
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_cloth_dict["7-Hair"]
    template['BODY'] = sam_setup

    # Torso definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_cloth_dict["0-Torso"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['TOP_GARMENT'] = sam_setup

    # Bottom garment definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["LLUF",0.25],["LRUF",0.25],["TORS",0.95],["LLUF",0.5],["LRUF",0.5]]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 1
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_cloth_dict["12-Trousers"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['BOTTOM_GARMENT'] = sam_setup

    # Head definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup['SI_POINTS'] = [["HEAD", 0.1]]
    sam_setup["SE_POINTS"] = [["ARLO", 0.5],["ALLO",0.5],["ARLI",0.5],["ARLO", 0.5]]
    sam_setup["BBOX"] = [["23-Head-Right","24-Head-Left"], [1.05, 1.7], None]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_cloth_dict["11-Head"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    sam_setup["STORE_MASK"] = True
    template['HEAD'] = sam_setup

    # Left Hand definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["HLLI", 0.5],["HLLO",0.5]]
    sam_setup["BBOX"] = [["4-Hand-Left"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_cloth_dict["14-Left-Hand"]
    sam_setup["ADD_BUFFER"] = 4
    sam_setup["FILL_HOLES"] = True
    template['LEFT_HAND'] = sam_setup

    # Right Hand definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["HRLI", 0.5],["HRLO",0.5]]
    sam_setup["BBOX"] = [["3-Hand-Right"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_cloth_dict["9-Right-Hand"]
    sam_setup["ADD_BUFFER"] = 4
    sam_setup["FILL_HOLES"] = True
    template['RIGHT_HAND'] = sam_setup
                
    # Left arm definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["ALLO", 0.15],["ALLO", 0.90],["HLLI",0.1],["HLLO",0.1]]
    sam_setup["SE_POINTS"] = [["ARLO", 0.90],["ARLO",0.5],["ARLI",0.5],["LLUF",0.1],["HEAD",0.5]]
    sam_setup["BBOX"] = [["15-Arm-Left-Upper-Inner", "17-Arm-Left-Upper-Outer","19-Arm-Left-Lower-Inner","21-Arm-Left-Lower-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_cloth_dict["1-Left-Arm"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['LEFT_ARM'] = sam_setup         

    # Right arm definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["ARLO", 0.15],["ARLO", 0.95],["HRLI",0.1],["HRLO",0.1]]
    sam_setup["SE_POINTS"] = [["ALLO", 0.90],["ALLO",0.5],["ALLI",0.5],["LRUF",0.1],["HEAD",0.5]]
    sam_setup["BBOX"] = [["16-Arm-Right-Upper-Inner", "18-Arm-Right-Upper-Outer","20-Arm-Right-Lower-Inner","22-Arm-Right-Lower-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = ccg.semantic_cloth_dict["2-Right-Arm"]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['RIGHT_ARM'] = sam_setup

    #Mask building order
    template['PROCESSING_ORDER'] = ["BODY", 
                                    "LEFT_ARM",
                                    "RIGHT_ARM",
                                    "HEAD",
                                    "LEFT_HAND",
                                    "RIGHT_HAND",
                                    "TOP_GARMENT",
                                    "BOTTOM_GARMENT"]
   
    return template        

# A template for CVTON 'image_parse' masks
# Coluors in this mask have been hardcoded as their providence is unknown
# from the CVTON code
def samTemplateImageParse():
    template = {}
    template['PRE_PROCESSOR'] = False
    template['USE_PRE_PROCESSOR_LIST'] = ['samHairTemplate']

    # What directories are required
    template['BASE_IMAGE_INPUT_PATH'] = "detectron2/data"
    template['BASE_IMAGE_INPUT_FORMATS'] = ["jpg","png"]

    template['ADDITIONAL_INPUT_PATHS'] = ["detectron2/results_cvton", "detectron2/results_cvton_dump"]
    template['ADDITIONAL_INPUT_FORMATS'] = [["png"],["json"]]
    template['INPUT_KEYS'] = ['0_BASE','1_ADD_IMAGE','2_JSON']
    template['INPUT_COLOURS'] = 'CVTON'
   
    template['OUTPUT_SIZE']   = [384, 512]
    template['OUTPUT_PATHS'] = ["data/viton/data/image-parse"]
    template['RENAME_OUTPUT_FILES'] = True

    # Setup Background
    template['BACKGROUND_TYPE'] = "SOLID_FILL"
    template['BACKGROUND_COLOUR'] = [0, 0, 0]
   
    # Top garment definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = [254, 85, 0]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['TOP_GARMENT'] = sam_setup

    # Bottom garment definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["LLUF",0.25],["LRUF",0.25],["TORS",0.95],["LLUF",0.5],["LRUF",0.5]]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 1
    sam_setup['OUTPUT_COLOUR'] =[85, 85, 0]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['BOTTOM_GARMENT'] = sam_setup

    # Head definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup['SI_POINTS'] = [["HEAD", 0.1]]
    sam_setup["SE_POINTS"] = [["ARLO", 0.5],["ALLO",0.5],["ARLI",0.5],["ARLO", 0.5]]
    sam_setup["BBOX"] = [["23-Head-Right","24-Head-Left"], [1.05, 1.7], None]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = [0, 0, 254]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    sam_setup["STORE_MASK"] = True
    template['HEAD'] = sam_setup

    # Left Hand definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["HLLI", 0.5],["HLLO",0.5]]
    sam_setup["BBOX"] = [["4-Hand-Left", "15-Arm-Left-Upper-Inner", "17-Arm-Left-Upper-Outer","19-Arm-Left-Lower-Inner","21-Arm-Left-Lower-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = [85, 254, 169]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['LEFT_HAND'] = sam_setup

    # Right Hand definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["HRLI", 0.5],["HRLO",0.5]]
    sam_setup["BBOX"] = [["3-Hand-Right","16-Arm-Right-Upper-Inner", "18-Arm-Right-Upper-Outer","20-Arm-Right-Lower-Inner","22-Arm-Right-Lower-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = [0, 254, 254]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['RIGHT_HAND'] = sam_setup
    
    #Mask building order
    template['PROCESSING_ORDER'] = ["HEAD",
                                    "LEFT_HAND",
                                    "RIGHT_HAND",
                                    "TOP_GARMENT",
                                    "BOTTOM_GARMENT"]
    
    template["POST_PROCESSING"] = {"ADD_NECK"  : {"OUT_COLOUR" : [0, 0, 0],
                                                  "THRESHOLD" : 0.65},
                                   "ADD_SAVED" : {"NAME" : "HAIR",
                                                  "OUT_COLOUR" : [255, 0, 0]}}

    template["POST_PROCESSING_ORDER"] = ["ADD_NECK", "ADD_SAVED"]

    return template  

# Template for extracting the top garment from an image 
# onto a white background
def samTemplateGarment():
    template = {}
    template['PRE_PROCESSOR'] = False

    # What directories are required
    template['BASE_IMAGE_INPUT_PATH'] = "detectron2/data"
    template['BASE_IMAGE_INPUT_FORMATS'] = ["jpg","png"]

    template['ADDITIONAL_INPUT_PATHS'] = ["detectron2/results_cvton", "detectron2/results_cvton_dump"]
    template['ADDITIONAL_INPUT_FORMATS'] = [["png"],["json"]]
    template['INPUT_KEYS'] = ['0_BASE','1_ADD_IMAGE','2_JSON']
    template['INPUT_COLOURS'] = 'CVTON'
   
    template['OUTPUT_SIZE']   = [384, 512]
    template['OUTPUT_PATHS'] = ["data/demos/cloth"]
    
    template['RENAME_OUTPUT_FILES'] = True

    # Setup Background
    template['BACKGROUND_TYPE'] = "SOLID_FILL"
    template['BACKGROUND_COLOUR'] = [255, 255, 255]
    
    # Torso definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"]  = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"]  = True
    sam_setup['MODE']       = "BASE_MASK"
    sam_setup["ADD_BUFFER"] = 0
    sam_setup["FILL_HOLES"] = True
    template['TOP_GARMENT'] = sam_setup

    #Mask building order
    template['PROCESSING_ORDER'] = ["TOP_GARMENT"]
    
    return template           

# Template for extracting the garment from an image as 
# a black and white binary mask
def samTemplateGarmentMask():
    template = {}
    template['PRE_PROCESSOR'] = False

    # What directories are required
    template['BASE_IMAGE_INPUT_PATH'] = "detectron2/data"
    template['BASE_IMAGE_INPUT_FORMATS'] = ["jpg","png"]

    template['ADDITIONAL_INPUT_PATHS'] = ["detectron2/results_cvton", "detectron2/results_cvton_dump"]
    template['ADDITIONAL_INPUT_FORMATS'] = [["png"],["json"]]
    template['INPUT_KEYS'] = ['0_BASE','1_ADD_IMAGE','2_JSON']
    template['INPUT_COLOURS'] = 'CVTON'
   
    template['OUTPUT_SIZE']   = [384, 512]
    template['OUTPUT_PATHS'] = ["data/demos/cloth-mask"]
    
    template['RENAME_OUTPUT_FILES'] = True

    # Setup Background
    template['BACKGROUND_TYPE'] = "SOLID_FILL"
    template['BACKGROUND_COLOUR'] = [0, 0, 0]
    
    # Torso definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"]  = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"]  = True
    sam_setup['OUTPUT_COLOUR'] = [255, 255, 255]
    sam_setup["ADD_BUFFER"] = 0
    sam_setup["FILL_HOLES"] = True
    template['TOP_GARMENT'] = sam_setup

    #Mask building order
    template['PROCESSING_ORDER'] = ["TOP_GARMENT"]
    
    return template

# Template for extractinbg an 'agnostic' image from the base
# image - i.e. an image with the garment removed and replaced
# with a 'grey blob'.
def samTemplateGarmentAgnostic32():
    template = {}
    template['PRE_PROCESSOR'] = False

    # What directories are required
    template['BASE_IMAGE_INPUT_PATH'] = "detectron2/data"
    template['BASE_IMAGE_INPUT_FORMATS'] = ["jpg","png"]

    template['ADDITIONAL_INPUT_PATHS'] = ["detectron2/results_cvton", "detectron2/results_cvton_dump"]
    template['ADDITIONAL_INPUT_FORMATS'] = [["png"],["json"]]
    template['INPUT_KEYS'] = ['0_BASE','1_ADD_IMAGE','2_JSON']
    template['INPUT_COLOURS'] = 'CVTON'
   
    template['OUTPUT_SIZE']   = [384, 512]
    template['OUTPUT_PATHS'] = ["data/demos/agnostic_3.2"]
    
    template['RENAME_OUTPUT_FILES'] = True

    # Setup Background
    template['BACKGROUND_TYPE'] = "0_BASE"
    template['BACKGROUND_COLOUR'] = [255, 255, 255]

    # Overall body definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 2
    sam_setup['MODE'] = "BASE_MASK"
    template['BODY'] = sam_setup

    # Torso definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"]  = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"]  = True
    sam_setup['OUTPUT_COLOUR'] = [128, 128, 128]
    sam_setup["ADD_BUFFER"] = 15
    sam_setup["FILL_HOLES"] = True
    template['TOP_GARMENT'] = sam_setup

    #Mask building order
    template['PROCESSING_ORDER'] = ["BODY", "TOP_GARMENT"]
    
    return template




           
