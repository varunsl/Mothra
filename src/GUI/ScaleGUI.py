# import the necessary packages
import cv2
from .BaseGUI import BaseGUI
from .tools import auto_tools

class ScaleGUI(BaseGUI):
    def __init__(self, img):
        instructions = '''
                First set the image in the correct orientation. Press 'V'
                to flip vertically and 'H' to flip horizontally.
                Then set the scale factor: draw a horizontal 1 cm long line.
                Press 'Z' to undo, or 'X' to continue.
                '''
        super().__init__(img, "Draw Scale Factor", instructions)
        
        self.v_flipped = False
        self.h_flipped = False

        # endpoitns of the drawn line segment
        self.pt1, self.pt2 = None, None
    
    def mouse_input(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # clear the image if a line has already been drawn
            if self.pt1 or self.pt2:
                self.image = self.clone.copy()
            self.pt1, self.pt2 = (x, y), None

        elif event == cv2.EVENT_LBUTTONUP:
            self.pt2 = (x, y)

            cv2.line(self.image, self.pt1, self.pt2, (0, 255, 0), 2)

    def generate_guess(self):
        if auto_tools.is_upside_down(self.image):
            self.v_flipped = not self.v_flipped
            self.image = cv2.flip(self.image, 0)
            self.clone = self.image.copy()
    
    def undo(self):
        # clear data
        self.pt1, self.pt2 = None, None
        self.image = self.clone.copy()

    def special_key_input(self, key):
        # don't allow user to move to previous step
        if key == ord(","):
            return True
        
        # flip vertically
        elif key == ord("v"):
            self.v_flipped = not self.v_flipped
            self.image = cv2.flip(self.clone, 0)
            self.clone = self.image.copy()
            self.pt1, self.pt2 = None, None
            return True
        
        # flip horizontally
        elif key == ord("h"):
            self.h_flipped = not self.h_flipped
            self.image = cv2.flip(self.clone, 1)
            self.clone = self.image.copy()
            self.pt1, self.pt2 = None, None
            return True
    
    def can_exit(self):
        # the step is considered complete if we have the endpoints of the segment
        return self.pt1 and self.pt2
    
    def extract_data(self):
        # this step returns the horizontal distance of the segment the user drew
        return self.pt1[0] - self.pt2[0]
