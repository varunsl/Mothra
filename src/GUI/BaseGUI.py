import numpy as np
import cv2
from enum import Enum

class ExitCode(Enum):
    '''
    This enum is used to keep track of whether a step is still running,
    has finished normally, or has been skipped.
    '''
    READY = 1
    DONE = 2
    NEXT = 3
    PREV = 4

class BaseGUI:
    '''
    This is an abstract class representing some step of the program.
    '''    
    def __init__(self, image, step_name, instructions):
        self.image = image.copy()
        self.clone = image.copy()

        self.step_name = step_name
        self.instructions = instructions

        self.exit_code = ExitCode.READY
        self.completed_once = False
    
    def setup(self):
        if not self.can_exit():
            self.generate_guess()
        
        cv2.namedWindow(self.step_name)
        cv2.moveWindow(self.step_name, 40, 30)
        cv2.setMouseCallback(self.step_name, self.mouse_input)

        print(self.instructions)

    def generate_guess(self):
        pass

    def display(self):
        '''Show the window'''
        cv2.imshow(self.step_name, self.image)
    
    def mouse_input(self, event, x, y, flags, param):
        '''Handles mouse input from the user.'''
        pass
    
    def key_input(self):
        key = cv2.waitKey(1) & 0xFF

        handled = self.special_key_input(key)

        if handled:
            return

        elif key == 27: # esc key
            raise SystemExit(0)
        
        elif key == ord("z"):
            self.undo()

        elif key == ord("x") and self.can_exit():
            self.exit_code = ExitCode.DONE
        
        elif key == ord(","):
            self.exit_code = ExitCode.PREV
        
        elif key == ord(".") and self.completed_once:
            self.exit_code = ExitCode.NEXT

    def special_key_input(self, key):
        # any keyboard input particular to the step is defined here
        # returns True if any action was taken
        pass

    def undo(self):
        pass
    
    def can_exit(self):
        # called to check if the step can be exited; most importantly,
        # checks if the data from the step has been extracted
        pass

    def exit(self):
        # called when the step is exited; does any necessary clean up
        cv2.destroyAllWindows()
    
    def extract_data(self):
        # returns the data that the step was supposed to calculate
        pass

    def run(self):
        self.setup()
        while self.exit_code == ExitCode.READY:
            self.display()
            self.key_input()
        self.exit()

if __name__ == '__main__':
    img = None
    gui = BaseGUI(img)
    gui.run()
