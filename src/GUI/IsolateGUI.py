from enum import Enum

import numpy as np
import cv2
from .BaseGUI import BaseGUI
from .tools import img_tools
from .tools import auto_tools

class IsolateGUI(BaseGUI):
    def __init__(self, img):
        instructions = ''''
            Remove background: erase the background with mouse.
            Use the left mouse button for the flood tool or the
            right mouse button to erase normally.
            Press 'Z' to undo, or 'X' to continue.
            '''''
        
        super().__init__(img, "Isolate Moth", instructions)

        self.actions = Enum('MouseActions', 'NOOP FLOOD ERASE')
        self.action = self.actions.NOOP
        
        self.edits = [self.image]
    
    def mouse_input(self, event, x, y, flags, param):
        # set state
        if event == cv2.EVENT_LBUTTONDOWN:
            self.action = self.actions.FLOOD
            self.edits.append(self.edits[-1].copy())
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.action = self.actions.ERASE
            self.edits.append(self.edits[-1].copy())
        elif (event == cv2.EVENT_LBUTTONUP) or (event == cv2.EVENT_RBUTTONUP):
            self.action = self.actions.NOOP

        # take action based on state
        if self.action == self.actions.FLOOD:
            self.edits[-1] = img_tools.flood_fill(self.edits[-1], (x, y))
        elif self.action == self.actions.ERASE:
            cv2.circle(self.edits[-1], (x, y), 3, (0,0,0), -1)

    def generate_guess(self):
        self.edits.append(img_tools.flood_remove(self.image))
    
    def display(self):
        cv2.imshow("Isolate Moth", self.edits[-1])
    
    def undo(self):
        # clear data
        if len(self.edits) > 1:
            self.edits.pop()
    
    def can_exit(self):
        # the step is considered complete if we have the endpoints of the segment
        return len(self.edits) > 1
    
    def extract_data(self):
        bgr = cv2.cvtColor(self.edits[-1],cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,15,255,cv2.THRESH_BINARY)

        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        
        #img_tools.convexity_defects(opening)
        auto_tools.auto_body(opening)
        #img_tools.show(opening)
        return opening
