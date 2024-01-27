# A set of lists and dictionaries containing semantic labels and colour mappings for
# the densepose and CVTON models 
# Dr Paul Goodman, NICD, Newcastle University, 27/01/2024

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

densepose_groupings = {"BACKG" : [0],
                       "BODY"  : [1, 2],
                       "R_ARM" : [3, 16, 18, 20, 22],
                       "L_ARM" : [4, 15, 17, 19, 21],
                       "R_LEG" : [6, 7 ,9, 11, 13],
                       "L_LEG" : [5, 8, 10, 12,14],
                       "HEAD"  : [23, 24]}

densepose_groupings_with_hands = {"BACKG" : [0],
                                  "BODY"  : [1, 2],
                                  "R_HAND": [3],
                                  "L_HAND": [4],
                                  "R_ARM" : [16, 18, 20, 22],
                                  "L_ARM" : [15, 17, 19, 21],
                                  "R_LEG" : [6, 7 ,9, 11, 13],
                                  "L_LEG" : [5, 8, 10, 12,14],
                                  "HEAD"  : [23,24]}

densepose_raw = {item: [index] for index, item in enumerate(densepose_semantic_labels)}

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

input_mode_dict = { "DEFAULT": densepose_semantic_cols,
                    "CVTON"  : cvton_semantic_cols }

group_mode_dict = { "DEFAULT"     : densepose_groupings,
                    "WITH_HANDS"  : densepose_groupings_with_hands,
                    "RAW"         : densepose_raw }