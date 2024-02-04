import copy
from itertools import accumulate
import pauls_utils as pu
import numpy as np
import matplotlib.pyplot as plt

def plotImageAndSkeleton(image, 
                         results=None, 
                         groupings=None, 
                         skeleton=None, 
                         control_points=None, 
                         win_title=None):
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    # Display the binary mask
    plt.subplot(1, 2, 2)
    h, w, _ = image.shape
    black_image = np.zeros((h, w, 3), dtype=np.uint8)
    plt.imshow(black_image)
    plt.title('OOBBs + Skeleton')
    if not results==None and len(results)>0:
        head_pt = results['HEAD_PT']
        #print(f"Val: {head_pt}")
        plt.scatter(head_pt[1], head_pt[0], marker='H', color='white', edgecolors='black', s=100)

        for body_part in groupings.keys():
            oobbs = results[body_part]['OOBBS']
            oobb_cents = results[body_part]['OOBB_CENTS']
            oobb_mids = results[body_part]['OOBB_MIDS']
            colour = results[body_part]['RGB']
            norm_rgb = [val / 255.0 for val in colour]
            cutoff = 2
            for i, oobb in enumerate(oobbs):
                if i>=cutoff:
                    break
                y1, x1, y2, x2, y3, x3, y4, x4 = [item for sublist in oobb for item in sublist]
                plt.plot([x1, x2], [y1, y2], linestyle='-', color=norm_rgb, label='Line 1')
                plt.plot([x2, x3], [y2, y3], linestyle='-', color=norm_rgb, label='Line 2')
                plt.plot([x3, x4], [y3, y4], linestyle='-', color=norm_rgb, label='Line 3')
                plt.plot([x4, x1], [y4, y1], linestyle='-', color=norm_rgb, label='Line 4') 
            for i, cent in enumerate(oobb_cents):
                if i>=cutoff:
                    break
                plt.scatter(cent[1], cent[0], color=norm_rgb, marker='*')
            for i, mids in enumerate(oobb_mids):
                if i>=cutoff:
                    break
                y1, x1, y2, x2, y3, x3, y4, x4 = [item for sublist in mids for item in sublist]
                plt.plot([x4, x1], [y4, y1], linestyle='--', color=norm_rgb, label='Line 1')
                plt.plot([x3, x2], [y3, y2], linestyle='--', color=norm_rgb, label='Line 2')           
    
    if not skeleton==None and len(skeleton)>0:
        for key, bone in skeleton.items():
            y1 = bone[0][0][0]
            x1 = bone[0][0][1]
            y2 = bone[0][1][0]
            x2 = bone[0][1][1]
            vis = bone[2]
            if vis == True:
                ls = '-'
                col = 'white'
            else:
                ls = '--'
                col = 'red'
            plt.plot([x1, x2], [y1, y2], linestyle=ls, color=col, label=key, lw=3)

    if not control_points==None and len(control_points)>0: 
        for pts in control_points:
            y1 = pts[0]
            x1 = pts[1]
            if pts[2]:
                col = 'yellow'
            else:
                col = 'magenta'
            plt.scatter(x1, y1, marker='x', color=col, s=150)    

    # Set window name
    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title(win_title)
    plt.show()

# Get rescaled coordinates for a single bone (if necessary)
# NB: Creates a copy of the original metrics and returns the altered copy
def skeletonGetRescaledBoneMetrics(metrics, new_w, new_h):
    new_metrics = copy.copy(metrics)
    if new_metrics[1] == True and new_w>0 and new_h>0:
        new_metrics[0][0][0] = (round(new_metrics[0][0][0]*new_h))
        new_metrics[0][0][1] = (round(new_metrics[0][0][1]*new_w))
        new_metrics[0][1][0] = (round(new_metrics[0][1][0]*new_h))
        new_metrics[0][1][1] = (round(new_metrics[0][1][1]*new_w))
        new_metrics[1]       = False
        new_metrics[3]       = pu.calcDistance(new_metrics[0][0], new_metrics[0][1])
        new_metrics[4]       = pu.getVector(new_metrics[0][0], new_metrics[0][1])
    return new_metrics

# Get rescaled skeleton at the image level
def skeletonGetRescaledBones(skeleton, new_w, new_h):
    new_skeleton = {}
    if len(skeleton)>0 and new_w > 0 and new_h > 0:
        #print(f"skeleton: {skeleton}")
        for bone_name, bone_data in skeleton.items():
            #print(f"xx {bone_data} xx")
            new_bone_data = skeletonGetRescaledBoneMetrics(bone_data, new_w, new_h)
            #print(f"xx {new_bone_data} xx")
        new_skeleton[bone_name] = new_bone_data
    return new_skeleton

def skeletonGetUsedBoneGroups(skeleton):
    bone_groups = set()
    for bone in skeleton.keys():
        bone_groups.add(bone[:4])
    return bone_groups

def getSortedBodyPartListsForKeyPoint(body_part, key_pt):
    sorted_combined_list = []    
    if len(body_part['OOBB_MIDS'])>0:
        #order the torso points by proximity to the head point
        close_cand_pts = [] 
        far_cand_pts   = []   
        for i, block in enumerate(body_part['OOBB_MIDS']):
            if len(block)>0:
                sort_list = pu.getClosestToPointList(block, key_pt)
                #print(f"Block: {i}, Candidate close points list: {sort_list}")
                close_cand_pts.append([i, sort_list[0]])
                sort_list = pu.getFurthestToPointList(block, key_pt)
                #print(f"Block: {i}, Candidate far points list: {sort_list}")
                far_cand_pts.append([i, sort_list[0]])
                
        #Append lists and sort on last parameter?
        combined_list = close_cand_pts + far_cand_pts     
        sorted_combined_list = sorted(combined_list, key=lambda x: x[1][2])
        #print(f"Sorted combined List: {sorted_combined_list}")

    return sorted_combined_list

def skeletonGetBoneGroupContiguousDict(skeleton, bone_group):
    temp_dict = {}
    if len(skeleton)>0:
        #Get candidate bone list
        bones = pu.getKeysContainingSubstring(skeleton, bone_group)
        #print(f"{bones} : {len(bones)}")
        if len(bones)>0:
            sorted_bones = sorted(bones, key=lambda x: int(x[4:]))        
            #Check that bones are linked 
            #print(f"{sorted_bones} : {len(sorted_bones)}")
            for i, bone in enumerate(sorted_bones):
                this_bone = skeleton[bone]
                temp_dict[bone] = this_bone
                #print(f"Add orig bone: {bone} : {this_bone}")

                if i > 1 and not i == (len(sorted_bones)-1):
                    if this_bone[1] == True:
                        threshold = 0.001
                    else:
                        threshold = 1.0
                
                    next_name = sorted_bones[i+1]
                    next_bone = skeleton[next_name]
                    end_this = this_bone[0][1]
                    start_next = next_bone[0][0]
                    length = pu.calcDistance(end_this, start_next)
                    if length > threshold:
                        #print(f"Bone {bone} did not end at bone {next_name} start [{end_this} - {start_next}]")
                        temp_name = "ADD+" + str(i)
                        bone_vec = pu.getVector(end_this, start_next)
                        temp_data = [[end_this, start_next], False, False, length, bone_vec]
                        temp_dict[temp_name] = temp_data
                        #print(f"Add add+ bone: {temp_name} : {temp_data}")

    return temp_dict

def skeletonGetBoneLengths(skeleton):
    total_length = 0
    length_list = []
    if len(skeleton)>0:
        total_length = sum(sublist[3] for sublist in skeleton.values() if len(sublist) >= 4)
        length_list = [sublist[3] for sublist in skeleton.values() if len(sublist) >= 4]
        length_list = list(accumulate(length_list))
        #print(f"Returned bone length  : {total_length}")
        #print(f"Returned chainage list: {length_list}")
    return total_length, length_list

def skeletonGetBoneControlPoint(skeleton, bone_group, t_val):
    control_point = []
    if len(skeleton)>0 and t_val>=0 and t_val<=1: 
        temp_dict = skeletonGetBoneGroupContiguousDict(skeleton, bone_group)
        bone_names = [key for key in temp_dict.keys()]
        total_bone_length, cum_list = skeletonGetBoneLengths(temp_dict)
        t_cp = t_val * total_bone_length
        #print(f"Calculated control point chainage: {t_cp}")
        if len(cum_list)>0:
            # Find in which bone the control point lies 
            idx = next((i for i, element in enumerate(cum_list) if element >= t_cp), None)
            bone_data = temp_dict[bone_names[idx]]
            r_pt = 1.0 - ((cum_list[idx] - t_cp) / bone_data[3])
            pt_y = bone_data[0][0][0] + (r_pt * bone_data[4][0] * bone_data[3])
            pt_x = bone_data[0][0][1] + (r_pt * bone_data[4][1] * bone_data[3])
            control_point = [pt_y, pt_x, bone_data[2]]
            #print(f"Control point lies in bone {bone_names[idx]} : {bone_data} at relative point {r_pt}, coordinates: {control_point}, visible: {visible}")
            
    return control_point

def skeletonGetBonesTorsoAndHead(results):
    new_bones = {}
    
    #get the closest torso part to the head point 
    body_part = results['2-Torso-Front'] 
    key_pt = results['HEAD_PT'] 

    pt_list = getSortedBodyPartListsForKeyPoint(body_part, key_pt)

    #First torso-head bone
    if (len(pt_list)>0):
        bone_st  = copy.copy(key_pt)
        bone_en  = copy.copy(pt_list[0][1][1])
        bone_vec = pu.getVector(bone_st, bone_en)
        bone_len = pu.calcDistance(bone_st, bone_en)
        new_bones['HEAD1'] = [[bone_st, bone_en], False, True, bone_len, bone_vec]
        bone_root = 'TORS'
        vis = True        
        for i, _ in enumerate(pt_list[:-1]):
            bone_st = copy.copy(pt_list[i][1][1])
            bone_en = copy.copy(pt_list[i+1][1][1])
            bone_len = pu.calcDistance(bone_st, bone_en)
            bone_vec = pu.getVector(bone_st, bone_en)
            bone_name = bone_root+str(i+1)
            new_bones[bone_name] = [[bone_st, bone_en], False, vis, bone_len, bone_vec]
            vis = not vis

    return new_bones

def skeletonGetShoulderBone(results, key_pt, bone_st, part_name, out_name):
    bone_name = bone_data = None
    if part_name in results.keys():
        body_part = results[part_name] 
        #print(f"{body_part['NAME']}")
        pt_list = getSortedBodyPartListsForKeyPoint(body_part, key_pt)
        if len(pt_list)>0:
            bone_st = copy.copy(bone_st)
            bone_en = copy.copy(pt_list[0][1][1])
            bone_vec = pu.getVector(bone_st, bone_en)
            bone_len = pu.calcDistance(bone_st, bone_en)
            bone_name = out_name
            bone_data = [[bone_st, bone_en], False, False, bone_len, bone_vec]
    return bone_name, bone_data

def skeletonGetShoulderBones(results):
    new_bones = {}
    body_part = results['2-Torso-Front'] 
    key_pt = results['HEAD_PT'] 
    pt_list = getSortedBodyPartListsForKeyPoint(body_part, key_pt)
    
    if len(pt_list)>0:
        bone_names  = ['SLUI1','SLUO1','SRUI1','SRUO1']
        body_parts  = ['15-Arm-Left-Upper-Inner', 
                       '17-Arm-Left-Upper-Outer', 
                       '16-Arm-Right-Upper-Inner', 
                       '18-Arm-Right-Upper-Outer']

        for name, part in zip(bone_names, body_parts):
            bone_st  = copy.copy(pt_list[0][1][1])
            key_pt = bone_st
            bone_name, bone_data = skeletonGetShoulderBone(results, key_pt, bone_st, part, name)
            if not bone_name == None:
                new_bones[bone_name] = bone_data

    return new_bones     

def skeletonGetLimbBones(results, key_pt, part_name, out_root):
    bone_names = []
    bone_data  = []    
    if part_name in results.keys():
        body_part = results[part_name] 
        #print(f"{body_part['NAME']}")
        pt_list = getSortedBodyPartListsForKeyPoint(body_part, key_pt)
        if (len(pt_list)>0):      
            vis = True
            for i, _ in enumerate(pt_list[:-1]):
                bone_st   = copy.copy(pt_list[i][1][1])
                bone_en   = copy.copy(pt_list[i+1][1][1])
                bone_vec  = pu.getVector(bone_st, bone_en)
                bone_len  = pu.calcDistance(bone_st, bone_en)
                bone_name = out_root+str(i+1)
                bone_info = [[bone_st, bone_en], False, vis, bone_len, bone_vec] 
                bone_names.append(bone_name)
                bone_data.append(bone_info)
                vis = not vis
    
    return bone_names, bone_data

def skeletonGetUpperArmBones(results, skeleton):
    new_bones  = {}
    key_names  = ['SLUI1','SLUO1','SRUI1','SRUO1']
    body_parts = ['15-Arm-Left-Upper-Inner', 
                  '17-Arm-Left-Upper-Outer', 
                  '16-Arm-Right-Upper-Inner', 
                  '18-Arm-Right-Upper-Outer']
    out_names  = ['ALUI','ALUO','ARUI','ARUO'] 
    zipped_list = list(zip(key_names, body_parts, out_names))

    for key_name, body_part, out_name in zipped_list:
        if key_name in skeleton.keys():
            key_pt = skeleton[key_name][0][1]
            bone_names, bone_data = skeletonGetLimbBones(results, key_pt, body_part, out_name)
            if len(bone_names)>0:
                for bone_name, bone in zip(bone_names, bone_data):
                    #print(f"Bone name: {bone_name} : {bone}")
                    new_bones[bone_name] = bone
        
    return new_bones

def skeletonGetLastBoneInSequence(skeleton, root_name):
    last_key = None
    candidates = pu.getKeysContainingSubstring(skeleton, root_name)
    if len(candidates)>0:
        last_key = max(candidates, key=lambda key: int(key[4:]))
    return last_key

def skeletonGetLowerArmBones(results, skeleton):   
    new_bones  = {}
    key_names  = ['ALUI','ALUO','ARUI','ARUO']
    body_parts = ['19-Arm-Left-Lower-Inner', 
                  '21-Arm-Left-Lower-Outer', 
                  '20-Arm-Right-Lower-Inner', 
                  '22-Arm-Right-Lower-Outer']
    out_names  = ['ALLI','ALLO','ARLI','ARLO']     

    zipped_list = list(zip(key_names, body_parts, out_names))

    for key_name, body_part, out_name in zipped_list:
        #print(f"Key name: {key_name}")
        last_bone = skeletonGetLastBoneInSequence(skeleton, key_name)
        if not last_bone==None:
            key_pt = copy.copy(skeleton[last_bone][0][1])
            bone_names, bone_data = skeletonGetLimbBones(results, key_pt, body_part, out_name)
            if len(bone_names)>0:
                for bone_name, bone in zip(bone_names, bone_data):
                    new_bones[bone_name] = bone
    
    return new_bones

def skeletonGetHands(results, skeleton):
    new_bones  = {}
    key_names  = ['ALLI','ALLO','ARLI','ARLO']
    body_parts = ['4-Hand-Left', 
                  '4-Hand-Left', 
                  '3-Hand-Right', 
                  '3-Hand-Right']
    out_names  = ['HLLI','HLLO','HRLI','HRLO']

    zipped_list = list(zip(key_names, body_parts, out_names))

    for key_name, body_part, out_name in zipped_list:
        last_bone = skeletonGetLastBoneInSequence(skeleton, key_name)
        if not last_bone==None:
            key_pt = copy.copy(skeleton[last_bone][0][1])
            bone_names, bone_data = skeletonGetLimbBones(results, key_pt, body_part, out_name)
            if len(bone_names)>0:
                for bone_name, bone in zip(bone_names, bone_data):
                    new_bones[bone_name] = bone

    return new_bones

def skeletonGetHipBones(results, skeleton):
    new_bones = {} 
    key_pt = skeletonGetBoneControlPoint(skeleton, "TORS", 0.8)
   
    if len(key_pt)>0:
        bone_names  = ['HLUF1','HRUF1']
        body_parts  = ['10-Leg-Left-Upper-Front', 
                       '9-Leg-Right-Upper-Front']

        for name, part in zip(bone_names, body_parts):
            bone_st = [key_pt[0], key_pt[1]]
            key_pt  = bone_st
            bone_name, bone_data = skeletonGetShoulderBone(results, key_pt, bone_st, part, name)
            if not bone_name == None:
                new_bones[bone_name] = bone_data

    return new_bones  

def skeletonGetUpperLegBones(results, skeleton):
    new_bones  = {}
    key_names  = ['HLUF1','HRUF1']
    body_parts = ['10-Leg-Left-Upper-Front', 
                  '9-Leg-Right-Upper-Front']
    out_names  = ['LLUF','LRUF'] 
    zipped_list = list(zip(key_names, body_parts, out_names))

    for key_name, body_part, out_name in zipped_list:
        if key_name in skeleton.keys():
            key_pt = skeleton[key_name][0][1]
            bone_names, bone_data = skeletonGetLimbBones(results, key_pt, body_part, out_name)
            if len(bone_names)>0:
                for bone_name, bone in zip(bone_names, bone_data):
                    new_bones[bone_name] = bone
        
    return new_bones

def skeletonGetLowerLegBones(results, skeleton):
    new_bones  = {}
    key_names  = ['LLUF','LRUF']
    body_parts = ['14-Leg-Left-Lower-Front',
                  '13-Leg-Right-Lower-Front']
    out_names  = ['LLLF','LRLF']     

    zipped_list = list(zip(key_names, body_parts, out_names))

    for key_name, body_part, out_name in zipped_list:
        #print(f"Key name: {key_name}")
        last_bone = skeletonGetLastBoneInSequence(skeleton, key_name)
        if not last_bone==None:
            key_pt = copy.copy(skeleton[last_bone][0][1])
            bone_names, bone_data = skeletonGetLimbBones(results, key_pt, body_part, out_name)
            if len(bone_names)>0:
                for bone_name, bone in zip(bone_names, bone_data):
                    new_bones[bone_name] = bone
    
    return new_bones

def skeletonGetFeet(results, skeleton):
    new_bones  = {}
    key_names  = ['LLLF','LRLF']
    body_parts = ['5-Foot-Left', 
                  '6-Foot-Right']
    out_names  = ['FLLF','FLRF']

    zipped_list = list(zip(key_names, body_parts, out_names))

    for key_name, body_part, out_name in zipped_list:
        last_bone = skeletonGetLastBoneInSequence(skeleton, key_name)
        if not last_bone==None:
            key_pt = copy.copy(skeleton[last_bone][0][1])
            bone_names, bone_data = skeletonGetLimbBones(results, key_pt, body_part, out_name)
            if len(bone_names)>0:
                for bone_name, bone in zip(bone_names, bone_data):
                    new_bones[bone_name] = bone

    return new_bones

# Normalise bone coordinates if necessary
def skeletonNormaliseBoneMetrics(metrics, w, h):
    new_metrics = copy.copy(metrics)
    if new_metrics[1]==False:
        #print(f"Init: [{metrics[0][0][0]} {metrics[0][0][1]}] - [{metrics[0][1][0]} {metrics[0][1][1]}]")        
        new_metrics[0][0][0] = metrics[0][0][0] / h
        new_metrics[0][0][1] = metrics[0][0][1] / w
        new_metrics[0][1][0] = metrics[0][1][0] / h
        new_metrics[0][1][1] = metrics[0][1][1] / w
        new_metrics[1]       = True
        #print(f"Points: {new_metrics[0][0]} {new_metrics[0][1]}")        
        new_metrics[3]       = pu.calcDistance(new_metrics[0][0], new_metrics[0][1])
        new_metrics[4]       = pu.getVector(new_metrics[0][0], new_metrics[0][1])
        #print(f"Fina: [{new_metrics[0][0][0]} {new_metrics[0][0][1]}] - [{new_metrics[0][1][0]} {new_metrics[0][1][1]}]")
    return new_metrics
