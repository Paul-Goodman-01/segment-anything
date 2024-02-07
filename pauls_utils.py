import os
import glob
import math
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
import traceback

# Calculate distance between two points - good 'ol Pythagoras
def calcDistance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def checkAndCreateDirectory(path, warn_if_present = False):
    OK = False
    if checkDirectory(path):
        print(f"Output directory '{path}' already exists!")
        if warn_if_present==True and os.listdir(path):
            print("WARNING! : files in {path} will be overwritten by outputs!")
        OK = True  
    else:
        try:
            os.makedirs(path)
            print(F"Directory '{path}' created successfully.")
            OK = True
        except OSError as e:
            print(f"Failed to create directory '{path}': {e}")
    return OK

# Check that required directories exist
def checkDirectory(path):
    return os.path.exists(path) and os.path.isdir(path)

# Check that tow objects are valid dictionaries and that their keys match
def doDictKeysMatch(dict1, dict2):
    return isValidDict(dict1) and isValidDict(dict2) and dict1.keys() == dict2.keys()

# Check that tow objects are valid dictionaries and that their keys match
def doListsMatch(list1, list2):
    return isValidList(list1) and isValidList(list2) and list1 == list2

# Dump stacktrace after an exception
def dumpException(e):
    print("Exception", e)
    stack_trace = traceback.format_exc()
    print("Stack Trace:")
    print(stack_trace)

# Simple function to print elements of a list
def dumpFileList(file_list):
    for i, file in enumerate(file_list):
        print(f"{i}. {file}")

# Order a list of points by their distance from another point - closest first 
def getClosestToPointList(pt_list, target_pt):
    results = []
    for i, pt in enumerate(pt_list):
        dist = calcDistance(pt, target_pt)
        results.append([i, pt, dist])
    
    #sort the list
    results = sorted(results, key=lambda x: x[2])

    return results

# Return the file title from a full path
def getFileTitle(file_path):
    file_name = os.path.basename(file_path)
    title, _ = os.path.splitext(file_name)
    return title

# Return a variable from a dictionary key, or return none
def getSafeDictKey(dictionary, keys):
    result = None
    try:
        if isValidDict(dictionary) and isValidList(keys):
            key = keys[0]
            if len(keys) == 1:
                return dictionary[key]
            else:
                return getSafeDictKey(dictionary[key], keys[1:])
    except (KeyError, TypeError):
        print("WARNING! : failed to get dictionary item from key list '{keys}'")
        pass
    return result
  
# Get a directory by a file dialog
def getDirectoryByDialog(win_title):
    root = tk.Tk() 
    root.withdraw()
    path = tk.filedialog.askdirectory( title = win_title )
    #print(f"Set path: '{path}'")
    return path 

# Get all files in a directory with extensions matching a filter list
def getFilesInDirectory(root, filters):
    file_list = []
    for ext in filters:
        pattern = f"{root}/*.{ext}"
        file_list.extend(glob.glob(pattern))
    return file_list

# Order a list of points by their distance from another point - closest last 
def getFurthestToPointList(pt_list, target_pt):
    results = getClosestToPointList(pt_list, target_pt)
    results.reverse()
    return results

# Get an image filename by opening a file dialog
def getImageFileByDialog():
    root = tk.Tk() 
    root.withdraw()
    image_path = askopenfilename( 
        title="Select a .PNG file",
        filetypes=[("PNG files", "*.png"),("JPG files", "*.jpg"),("All files", "*.*")]
    ) 
    return image_path

# Get all keys from a dictionary containing a substring
def getKeysContainingSubstring(dictionary, substring):
    results = []
    if len(dictionary)>0:
        results = [key for key in dictionary.keys() if substring in key]
    return results

# Get files in one directory whose titles contain the names of the file in the other directory
def getMatchingFileList(base_path, add_list):
    matching_add_files = None
    #print(f"Base list: {base_list}")
    #print(f"Add list : {add_list}")

    if isValidList(add_list):
        base_name = os.path.basename(base_path)
        file_name, _ = os.path.splitext(base_name)
        file_name = "/"+file_name+"."
        matching_add_files = [element for element in add_list if file_name in element]

        if not isValidList(matching_add_files):
            print(f"WARNING! : No matching additional files found in '{os.path.dirname(add_list[0])}'")
        #else:
        #    print(f"Found {len(matching_add_files)} in '{os.path.dirname(add_list[0])}'")
    
    else:
        print(f"WARNING! : Could not check lists for matching files!")

    return matching_add_files[0]


# Get a vector from two points {stored as 2D Lists}
def getVector(pt1, pt2):
    result = []
    e0 = pt2[0]-pt1[0]
    e1 = pt2[1]-pt1[1]

    #get vector length 
    len = calcDistance(pt1, pt2)  
    if len > 0.0:
        e0 = e0 / len
        e1 = e1 / len
        result = [e0, e1]

    return result

def getVectorDotProduct(v1, v2):
    result = v1[0]*v2[0] + v1[1]*v2[1]
    return result

def isValidDict(my_dict):
    if not isinstance(my_dict, dict):
        return False
    elif len(my_dict)==0:  # Empty dict
        return False
    else:
        return True

def isValidList(my_list):
    if not isinstance(my_list, list):
        return False
    elif len(my_list)==0:  # Empty list
        return False
    else:
        return True

def isValidNpArray(my_list):
    if not isinstance(my_list, np.ndarray):
        return False
    elif len(my_list)==0:  # Empty list
        return False
    else:
        return True