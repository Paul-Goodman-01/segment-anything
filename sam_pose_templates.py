

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
    result['ADD_BUFFER'] = None
    result['FILL_HOLES'] = False
    result['OUTPUT_COLOUR'] = [0, 0, 0]
    result['STORE_MASK'] = False
    return result

semantic_cloth_labels = [
        [128, 0, 128], # Torso/Top/Clothing\n",
        [128, 128, 64], # Left arm\n",
        [128, 128, 192], # Right arm\n",
        [0, 255, 0], # Neck\n",
        [0, 128, 128], # Dress\n",
        [128, 128, 128], # Something upper?\n",
        [0, 0, 0], # Background\n",
        [0, 128, 0], # Hair\n",
        [0, 64, 0], # Left leg?\n",
        [128, 128, 0], # Right hand\n",
        [0, 192, 0], # Left foot\n",
        [128, 0, 192], # Head\n",
        [0, 0, 192], # Legs / skirt?\n",
        [0, 64, 128], # Skirt?\n",
        [128, 0, 64], # Left hand\n",
        [0, 192, 128], # Right foot\n",
        [0, 0, 128],
        [0, 128, 64],
        [0, 0, 64],
        [0, 128, 192]]

def samBodyTemplateWithHands():
    template = {}

    # What directories are required
    template['BASE_IMAGE_INPUT_PATH'] = "detectron2/data"
    template['BASE_IMAGE_INPUT_FORMATS'] = ["jpg","png"]

    template['ADDITIONAL_INPUT_PATHS'] = ["detectron2/results_cvton", "detectron2/results_cvton_dump"]
    template['ADDITIONAL_INPUT_FORMATS'] = [["png"],["json"]]
    template['INPUT_KEYS'] = ['0_BASE','1_ADD_IMAGE','2_JSON']
    template['INPUT_COLOURS'] = 'CVTON'
    template['OUTPUT_PATHS'] = ["data/viton/data/image_parse_with_hands",
                                "data/viton/data/images",
                                "data/viton/data/image_densepose_parse",
                                "data/viton/data/pose"]
    template['RENAME_OUTPUT_FLIES'] = True

    # Setup Background
    template['BACKGROUND_TYPE'] = "SOLID_FILL"
    template['BACKGROUND_COLOUR'] = [0, 0, 0]
   
    # Overall body definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 2
    sam_setup['OUTPUT_COLOUR'] = [0, 128, 0]
    template['BODY'] = sam_setup

    # Torso definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["TORS", 0.3]]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = [128, 0, 128]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['TOP_GARMENT'] = sam_setup

    # Bottom garment definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["LLUF",0.25],["LRUF",0.25],["TORS",0.95],["LLUF",0.5],["LRUF",0.5]]
    sam_setup["MULTI_OUT"] = True
    sam_setup["FORCE_MULTI"] = 1
    sam_setup['OUTPUT_COLOUR'] =[0, 0, 192]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['BOTTOM_GARMENT'] = sam_setup

    # Head definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup['SI_POINTS'] = [["HEAD", 0.1]]
    sam_setup["SE_POINTS"] = [["ARLO", 0.5],["ALLO",0.5],["ARLI",0.5],["ARLO", 0.5]]
    sam_setup["BBOX"] = [["23-Head-Right","24-Head-Left"], [1.05, 1.7], None]
    sam_setup["MULTI_OUT"] = True
    sam_setup['OUTPUT_COLOUR'] = [128, 0, 192]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    sam_setup["STORE_MASK"] = True
    template['HEAD'] = sam_setup

    # Left Hand definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["HLLI", 0.5],["HLLO",0.5]]
    sam_setup["BBOX"] = [["4-Hand-Left"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = [128, 0, 64]
    sam_setup["ADD_BUFFER"] = 4
    sam_setup["FILL_HOLES"] = True
    template['LEFT_HAND'] = sam_setup

    # Right Hand definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["HRLI", 0.5],["HRLO",0.5]]
    sam_setup["BBOX"] = [["3-Hand-Right"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = [128, 128, 0]
    sam_setup["ADD_BUFFER"] = 4
    sam_setup["FILL_HOLES"] = True
    template['RIGHT_HAND'] = sam_setup
                
    # Left arm definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["ALLO", 0.15],["ALLO", 0.90],["HLLI",0.1],["HLLO",0.1]]
    sam_setup["SE_POINTS"] = [["ARLO", 0.90],["ARLO",0.5],["ARLI",0.5],["LLUF",0.1],["HEAD",0.5]]
    sam_setup["BBOX"] = [["15-Arm-Left-Upper-Inner", "17-Arm-Left-Upper-Outer","19-Arm-Left-Lower-Inner","21-Arm-Left-Lower-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = [128, 128, 64]
    sam_setup["ADD_BUFFER"] = 2
    sam_setup["FILL_HOLES"] = True
    template['LEFT_ARM'] = sam_setup         

    # Right arm definition
    sam_setup = getDefaultSAMInputDictionary()
    sam_setup["SI_POINTS"] = [["ARLO", 0.15],["ARLO", 0.95],["HRLI",0.1],["HRLO",0.1]]
    sam_setup["SE_POINTS"] = [["ALLO", 0.90],["ALLO",0.5],["ALLI",0.5],["LRUF",0.1],["HEAD",0.5]]
    sam_setup["BBOX"] = [["16-Arm-Right-Upper-Inner", "18-Arm-Right-Upper-Outer","20-Arm-Right-Lower-Inner","22-Arm-Right-Lower-Outer"], [1.05, 1.05], None]
    sam_setup['OUTPUT_COLOUR'] = [128, 128, 192]
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
    
    template["POST_PROCESSING"] = ["ADD_NECK"]

    return template        

    





           
