# import the necessary packages
import cv2
from .BaseGUI import BaseGUI

class WingHingeGUI(BaseGUI):
    def __init__(self, img, which_wing):
        instructions = '''
                Click the two points where the {} connects to the body.
                '''.format(which_wing)
        
        super().__init__(img, "Select Wing Hinge Points", instructions)
        self.pt1, self.pt2 = None, None
    
    def mouse_input(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # clear the image if a line has already been drawn
            if self.pt1 and self.pt2:
                self.undo()
                self.pt1 = (x, y)
            elif self.pt1:
                self.pt2 = (x,y)
            else:
                self.pt1 = (x, y)

            cv2.circle(self.image, (x, y), 3, (255, 0, 0), -1)            

    def generate_guess(self):
        pass
    
    def undo(self):
        # clear data
        self.pt1, self.pt2 = None, None
        self.image = self.clone.copy()

    def can_exit(self):
        # the step is considered complete if we have the endpoints of the segment
        return self.pt1 and self.pt2
    
    def extract_data(self):
        # this step returns the horizontal distance of the segment the user drew
        return self.pt1, self.pt2
