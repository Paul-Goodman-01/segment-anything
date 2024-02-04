import glob
import math
import tkinter as tk
from tkinter.filedialog import askopenfilename
import traceback

# Calculate distance between two points - good 'ol Pythagoras
def calcDistance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

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

# Get all keys from a dictionary containing a substring
def getKeysContainingSubstring(dictionary, substring):
    results = []
    if len(dictionary)>0:
        results = [key for key in dictionary.keys() if substring in key]
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