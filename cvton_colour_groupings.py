# A set of lists and dictionaries containing semantic labels and colour mappings for
# the densepose and CVTON models 
# Dr Paul Goodman, NICD, Newcastle University, 27/01/2024

densepose_semantic_cols = [(0, 0, 0), # 0, Background
	                       (25, 42, 114), # 1 Torso 1 (Rear)
	                       (14, 56, 135), # 2 Torso 2 (Front)
	                       (2, 67, 156), # 3 Right Hand
	                       (5, 77, 154), # 4 Left Hand
	                       (9, 85, 151), # 5 Left Foot
	                       (14, 92, 148), # 6 Right Foot
	                       (9, 100, 145), # 7 Right Leg Upper (Rear)
	                       (6, 109, 142), # 8 Left Leg Upper (Rear)
	                       (4, 116, 138), # 9 Right leg Upper (Front)
	                       (15, 120, 129), # 10 Left leg Upper (Front)
	                       (25, 125, 120), # 11 Right leg Lower (Rear)
	                       (39, 129, 110), # 12 Left Leg Lower (Rear)
	                       (60, 130, 100), # 13 Right leg Lower (Front)
	                       (79, 132, 92), # 14 Left Leg Lower (Front)
	                       (101, 133, 81), # 15 Left Arm - Upper inner
	                       (118, 132, 73), # 16 Right Arm - Upper inner
	                       (134, 131, 67), # 17 Left Arm - Upper outer
	                       (151, 130, 60), # 18 Right Arm - Upper outer
	                       (159, 134, 51), # 19 Left Arm - Lower inner
	                       (168, 139, 42), # 20 Right arm - Lower inner
	                       (176, 144, 32), # 21 Left Arm - Lower outer 
	                       (175, 154, 25), # 22 Right Arm - Lower outer
	                       (175, 164, 17), # 23 Right face
	                       (174, 175, 9)] # 24 Left Face  

densepose_semantic_labels = [ "0-Background",
	                          "1-Torso-Rear",
                              "2-Torso-Front",
                              "3-Hand-Right",
                              "4-Hand-Left",
                              "5-Foot-Left",
                              "6-Foot-Right",
                              "7-Leg-Right-Upper-Rear",
                              "8-Leg-Left-Upper-Rear",
                              "9-Leg-Right-Upper-Front",
                              "10-Leg-Left-Upper-Front",
                              "11-Leg-Right-Lower-Rear",
                              "12-Leg-Left-Lower-Rear",
                              "13-Leg-Right-Lower-Front",
                              "14-Leg-Left-Lower-Front",
                              "15-Arm-Left-Upper-Inner",
                              "16-Arm-Right-Upper-Inner",
                              "17-Arm-Left-Upper-Outer",
                              "18-Arm-Right-Upper-Outer",
                              "19-Arm-Left-Lower-Inner",
                              "20-Arm-Right-Lower-Inner",
                              "21-Arm-Left-Lower-Outer",
                              "22-Arm-Right-Lower-Outer",
                              "23-Head-Right",
                              "24-Head-Left"]

densepose_semantic_dict = dict(zip(densepose_semantic_labels, densepose_semantic_cols))

# Colours seem to be derived from LIP Challenge Data?
semantic_cloth_colours = [ [128, 0, 128],   # 0-Torso
                           [128, 128, 64],  # 1-Left arm
                           [128, 128, 192], # 2-Right arm
                           [0, 255, 0],     # 3-Neck
                           [0, 128, 128],   # 4-Dress
                           [128, 128, 128], # 5-Unknown-upper
                           [0, 0, 0],       # 6-Background
                           [0, 128, 0],     # 7-Hair
                           [0, 64, 0],      # 8-Left-leg
                           [128, 128, 0],   # 9-Right-hand
                           [0, 192, 0],     # 10-Left foot
                           [128, 0, 192],   # 11-Head
                           [0, 0, 192],     # 12-Trousers 
                           [0, 64, 128],    # 13-Skirt
                           [128, 0, 64],    # 14-Left-hand
                           [0, 192, 128],   # 15-Right-foot
                           [0, 0, 128],     # 16-Unknown-1
                           [0, 128, 64],    # 17-Unknown-2
                           [0, 0, 64],      # 18-Unknown-3
                           [0, 128, 192] ]  # 19-Unknown-4 

semantic_cloth_labels = [ "0-Torso",
                          "1-Left-Arm",
                          "2-Right-Arm",
                          "3-Neck",
                          "4-Dress",
                          "5-Unknown-upper",
                          "6-Background",
                          "7-Hair",
                          "8-Left-Leg",
                          "9-Right-Hand",
                          "10-Left-Foot",
                          "11-Head",
                          "12-Trousers", 
                          "13-Skirt",
                          "14-Left-Hand",
                          "15-Right-Foot",
                          "16-Unknown-1",
                          "17-Unknown-2",
                          "18-Unknown-3",
                          "19-Unknown-42" ]

semantic_cloth_dict = dict(zip(semantic_cloth_labels, semantic_cloth_colours))

# Unknown providence?
semantic_body_colours = [ [127, 127, 127], # 0-Head\n",
                          [0, 255, 255],   # 1-Left Hand\n",
                          [255, 255, 0],   # 2-Left-Arm-Upper",
                          [127, 127, 0],   # 3-Right-Hand\n",
                          [255, 127, 127], # 4-Unknown-1
                          [0, 255, 0],     # 5-Left-Leg",
                          [0, 0, 0],       # 6-Background",
                          [255, 127, 0],   # 7-Right-Arm-Upper",
                          [0, 0, 255],     # 8-Right-Leg",
                          [127, 255, 127], # 9-Unknown-2
                          [0, 127, 255],   # 10-Unknown-3
                          [127, 0, 255],   # 11-Right-Arm-Lower",
                          [255, 255, 127], # 12-Unknown-4
                          [255, 0, 0],     # 13-Torso",
                          [255, 0, 255] ]  # 14-Left-Arm-Lower"

semantic_body_labels = [ "0-Head",
                         "1-Left-Hand",
                         "2-Left-Arm-Upper",
                         "3-Right-Hand",
                         "4-Unknown-1",
                         "5-Left-Leg",
                         "6-Background",
                         "7-Right-Arm-Upper",
                         "8-Right-Leg",
                         "9-Unknown-2",
                         "10-Unknown-3",
                         "11-Right-Arm-Lower",
                         "12-Unknown-4",
                         "13-Torso",
                         "14-Left-Arm-Lower" ]

semantic_body_dict = dict(zip(semantic_body_labels, semantic_body_colours))

densepose_groupings = {"BACKG" : [0],
                       "BODY"  : [1, 2],
                       "R_ARM" : [3, 16, 18, 20, 22],
                       "L_ARM" : [4, 15, 17, 19, 21],
                       "R_LEG" : [6, 7 ,9, 11, 13],
                       "L_LEG" : [5, 8, 10, 12,14],
                       "HEAD"  : [23, 24]}

densepose_groupings_skeleton = { "BACKG" : [0], #0
                       			 "BODY"  : [6], #1
                       			 "R_ARM" : [6], #2
                       			 "L_ARM" : [6], #3
                       			 "R_LEG" : [1], #4
                       			 "L_LEG" : [1], #5
                       			 "HEAD"  : [6]} #6

densepose_groupings_with_hands = {"BACKG" : [0],
                                  "BODY"  : [1, 2],
                                  "R_HAND": [3],
                                  "L_HAND": [4],
                                  "R_ARM" : [16, 18, 20, 22],
                                  "L_ARM" : [15, 17, 19, 21],
                                  "R_LEG" : [6, 7 ,9, 11, 13],
                                  "L_LEG" : [5, 8, 10, 12,14],
                                  "HEAD"  : [23,24]}

densepose_groupings_with_hands_skeleton = {"BACKG" : [0], #0
                                  		   "BODY"  : [8], #1
                                  		   "R_HAND": [4], #2
                                  		   "L_HAND": [5], #3
                                  		   "R_ARM" : [8], #4
                                  		   "L_ARM" : [8], #5
                                  		   "R_LEG" : [1], #6
                                  		   "L_LEG" : [1], #7
                                  		   "HEAD"  : [8]} #8

densepose_raw = {item: [index] for index, item in enumerate(densepose_semantic_labels)}

densepose_raw_skeleton = {"0-Background" 				: [0],
	                  	  "1-Torso-Rear" 				: [23, 24],
                          "2-Torso-Front"				: [23, 24],
                          "3-Hand-Right" 				: [20, 22],
                          "4-Hand-Left"  				: [19, 21], 
                          "5-Foot-Left"  				: [12, 14],
                          "6-Foot-Right" 				: [11, 13],
                          "7-Leg-Right-Upper-Rear" 		: [1, 2],
                          "8-Leg-Left-Upper-Rear"  		: [1, 2],
                          "9-Leg-Right-Upper-Front" 	: [1, 2],
                          "10-Leg-Left-Upper-Front" 	: [1, 2],
                          "11-Leg-Right-Lower-Rear" 	: [7, 9],
                          "12-Leg-Left-Lower-Rear" 		: [8, 10],
                          "13-Leg-Right-Lower-Front" 	: [7, 9],
                          "14-Leg-Left-Lower-Front" 	: [8, 10],
                          "15-Arm-Left-Upper-Inner" 	: [23, 24],
                          "16-Arm-Right-Upper-Inner" 	: [23, 24],
                          "17-Arm-Left-Upper-Outer" 	: [23, 24],
                          "18-Arm-Right-Upper-Outer" 	: [23, 24],
                          "19-Arm-Left-Lower-Inner" 	: [15, 17],
                          "20-Arm-Right-Lower-Inner" 	: [16, 18],
                          "21-Arm-Left-Lower-Outer" 	: [15, 17],
                          "22-Arm-Right-Lower-Outer" 	: [16, 18],
                          "23-Head-Right" 				: [23],
                          "24-Head-Left" 				: [24] }

cvton_semantic_cols = [(0, 0, 0), # 0 Background
	                   (105, 105, 105), # 1 Torso 1 (Rear)
	                   (85, 107, 47), # 2 Torso 2 (Front)
	                   (139, 69, 19), # 3 Right Hand
	                   (72, 61, 139), # 4 Left Hand
	                   (0, 128, 0), # 5 Left Foot
	                   (154, 205, 50), # 6 Right Foot
	                   (0, 0, 139), # 7 Right Leg Upper (Rear)
                       (255, 69, 0), # 8 Left Leg Upper (Rear)
                       (255, 165, 0), # 9 Right leg Upper (Front)
	                   (255, 255, 0), # 10 Left leg Upper (Front)
	                   (0, 255, 0),  # 11 Right leg Lower (Rear)
	                   (186, 85, 211), # 12 Left Leg Lower (Rear)
	                   (0, 255, 127), # 13 Right leg Lower (Front)
                       (220, 20, 60), # 14 Left Leg Lower (Front)
	                   (0, 191, 255), # 15 Left Arm - Upper inner
	                   (0, 0, 255), # 16 Right Arm - Upper inner
	                   (216, 191, 216), # 17 Left Arm - Upper outer
	                   (255, 0, 255), # 18 Right Arm - Upper outer
	                   (30, 144, 255), # 19 Left Arm - Lower inner
	                   (219, 112, 147), # 20 Right arm - Lower inner
	                   (240, 230, 140), # 21 Left Arm - Lower outer 
	                   (255, 20, 147), # 22 Right Arm - Lower outer
	                   (255, 160, 122), # 23 Right face
	                   (127, 255, 212)] # 24 Left Face

semantic_cvton_dict = dict(zip(densepose_semantic_labels, cvton_semantic_cols))

input_mode_dict = { "DEFAULT": densepose_semantic_cols,
                    "CVTON"  : cvton_semantic_cols }

group_mode_dict = { "DEFAULT"     : densepose_groupings,
                    "WITH_HANDS"  : densepose_groupings_with_hands,
                    "RAW"         : densepose_raw }

skeleton_mode_dict = { "DEFAULT"    : densepose_groupings_skeleton, 
                       "WITH_HANDS" : densepose_groupings_with_hands_skeleton,
                       "RAW"		: densepose_raw_skeleton }