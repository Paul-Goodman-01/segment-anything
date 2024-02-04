import cv2
import numpy as np

# Example set of points
def maskToPoints(mask):
    # Find white pixels in the mask
    white_pixels = np.column_stack(np.where(mask > 0))
    return white_pixels

def getObjectOrientedBoundingBox(mask):
    points = np.array(maskToPoints(mask), dtype=np.float32)
    rect = cv2.minAreaRect(points)
    box_vertices = cv2.boxPoints(rect).astype(int)
    box_vertices = [[x, y] for y, x in box_vertices]
    return box_vertices

image = cv2.imread("test_mask.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

oobb = getObjectOrientedBoundingBox(image)
oobb = np.array(oobb)

cv2.drawContours(image, [oobb], 0, 255, 2)
for i, pt in enumerate(oobb):
    cv2.circle(image,(pt[0],pt[1]), 7*(i+1), (255,255,255), -1)
    # Display the result
cv2.imshow('Oriented Bounding Box', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



