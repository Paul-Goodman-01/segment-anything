import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename

def display_image_with_cursor(image_path):
    """
    Display an image in a separate window and allow the user to obtain RGB components at a particular pixel.

    Parameters:
    - image_path (str): Path to the image.

    Returns:
    - None
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    outputs = []

    if image is not None:
        image = cv2.resize(image, (768, 1024))
        window_name = image_path
        cv2.namedWindow(window_name)

        def on_mouse_event(event, x, y, flags, param):
            """
            Callback function for mouse events.

            Parameters:
            - event: The mouse event (cv2.EVENT_...).
            - x (int): The x-coordinate of the mouse cursor.
            - y (int): The y-coordinate of the mouse cursor.
            - flags: Any flags associated with the event.
            - param: Additional parameters.

            Returns:
            - None
            """
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get RGB components at the clicked pixel
                pixel_color = image[y, x]
                s_x = x / 768
                s_y = y / 1024
                pixel_data = (x, y, s_x, s_y, pixel_color)
                outputs.append(pixel_data)
                len_out = len(outputs)
                print(f"{len_out} : RGB at pixel ({x}, {y}): {pixel_color}")

        cv2.setMouseCallback(window_name, on_mouse_event)
        cv2.imshow(window_name, image)
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            
            key = cv2.waitKey(1) & 0xFF

            # Break the loop when the 'Esc' key is pressed
            if key == 27:
                break

        cv2.destroyAllWindows()
        return outputs
    else:
        print(f"Unable to load the image at {image_path}")

# Call open file dialog
root = tk.Tk() 
root.withdraw()
image_path = askopenfilename() 
print(f"Image path: '{image_path}'")

# Display the image in the cv2 window
outputs=[]
if not image_path==None and len(image_path)>0:
    outputs = display_image_with_cursor(image_path)

print(outputs)