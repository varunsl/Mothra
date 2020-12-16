# import the necessary packages
import cv2

from .tools import img_tools
from .tools import auto_tools
from .BaseGUI import BaseGUI

class CropGUI(BaseGUI):
    def __init__(self, img):
        instructions = '''
            Isolate moth: draw a bounding box.
            Press 'Z' to undo, or 'X' to continue.
            '''
        super().__init__(img, "Crop Moth", instructions)
        
        self.pt1, self.pt2 = None, None
    
    def mouse_input(self, event, x, y, flags, param):
        # TODO: Adjust bounding box by dragging
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # clear the image if a line has already been drawn
            if self.pt1:
                self.image = self.clone.copy()
            self.pt1, self.pt2 = (x, y), None

        elif event == cv2.EVENT_LBUTTONUP:
            self.pt2 = (x, y)

            # draw a rectangle where the user indicated
            cv2.rectangle(self.image, self.pt1, self.pt2, (0, 255, 0), 2)

    def generate_guess(self):
        self.pt1, self.pt2 = auto_tools.auto_crop(self.clone)
        cv2.rectangle(self.image, self.pt1, self.pt2, (0, 255, 0), 2)

    def undo(self):
        # clear data
        self.pt1, self.pt2 = None, None
        self.image = self.clone.copy()
    
    def can_exit(self):
        # the step is considered complete if we have the endpoints of the bounding box
        return self.pt1 and self.pt2
    
    def extract_data(self):
        # this step returns the cropped region of interest
        return (self.pt1[0], self.pt1[1], self.pt2[0] - self.pt1[0], self.pt2[1] - self.pt1[1])
