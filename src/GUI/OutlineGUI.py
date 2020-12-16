# import the necessary packages
import cv2
from .tools import img_tools
from .BaseGUI import BaseGUI

class OutlineGUI(BaseGUI):
    def __init__(self, img, mask, wing_points):
        instructions = '''
                Adjust the outline of the forewing.
                '''
        
        super().__init__(img, "Forewing Outline", instructions)
        self.mask = mask
        self.pt1, self.pt2 = wing_points
        self.outline = list()
        img_tools.convexity_defects_1(self.mask)
        pts = img_tools.convexity_defects(self.mask)
        f = lambda x: x[0] > max(self.pt1[0], self.pt2[0]) + 10
        pt = [x for x in pts if f(x)][0]
        cv2.circle(self.image,pt,3,[0,255,255],-1)
        cv2.circle(self.image,self.pt1,3,[0,0,255],-1)
        cv2.circle(self.image,self.pt2,3,[0,0,255],-1)
        img_tools.show(self.image)
    
    def mouse_input(self, event, x, y, flags, param):
        pass

    def generate_guess(self):
        img_tools.convexity_defects(self.mask)
    
    def undo(self):
        # clear data
        pass
        
    def can_exit(self):
        # the step is considered complete if we have the endpoints of the segment
        return True
    
    def extract_data(self):
        # this step returns the horizontal distance of the segment the user drew
        return None
