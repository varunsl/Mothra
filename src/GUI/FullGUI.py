import copy

from .BaseGUI import ExitCode
from .ScaleGUI import ScaleGUI
from .CropGUI import CropGUI
from .IsolateGUI import IsolateGUI
from .WingHingeGUI import WingHingeGUI
from .OutlineGUI import OutlineGUI

class FullGUI():
    '''
    This class holds all the steps of the program in a linked list
    structure.
    '''
    NUM_STEPS = 4
    
    def __init__(self, img):
        self.img = img
        
        self.steps = list()
        self.steps_backup = list()
        self.index = 0
        
        self.data = {}
        self.exit_code = ExitCode.READY
    
    @property
    def current(self):
        '''Return the current step.'''
        if self.index == len(self.steps):
            next_step = self.create_step(self.index)
            self.steps.append(next_step)
            self.backup_step(self.index)
        return self.steps[self.index]
    
    def next(self):
        '''Move to the next step.'''
        self.current.exit_code = ExitCode.READY
        
        self.index += 1
        
        if self.index == self.NUM_STEPS:
            # This exit_code is for the FullGUI
            self.exit_code = ExitCode.DONE
        
    def prev(self):
        '''Move to the previous step.'''
        self.current.exit_code = ExitCode.READY
        
        if self.index > 0:
            self.index -= 1

    def clear(self, index):
        '''Clear the steps from the index and onwards.'''
        self.steps = self.steps[:index]

    def create_step(self, index):
        if index == 0:
            return CropGUI(self.img)
        elif index == 1:
            return IsolateGUI(self.data['cropped_moth'])
        elif index == 2:
            return WingHingeGUI(self.data['cropped_moth'], 'forewing')
        elif index == 3:
            return OutlineGUI(self.data['cropped_moth'], self.data['moth_mask'], self.data['forewing_hinge_points'])

    def backup_step(self, index):
        backup = copy.deepcopy(self.steps[index])
        if index == len(self.steps_backup):
            self.steps_backup.append(backup)
        else:
            self.steps_backup[index] = backup

    def restore_step(self, index):
        self.steps[index] = self.steps_backup[index]
        self.backup_step(index)

    def update_data(self, index):
        # also write data when finished so the app can pick up where
        # it left off if quit suddenly
        data_to_add = self.steps[index].extract_data()
        if index == -1:
            self.data['scale_factor'] = data_to_add
            self.data['oriented_image'] = self.steps[index].clone
        elif index == 0:
            x, y, w, h = data_to_add
            self.data['cropped_moth'] = self.img[y:y+h, x:x+w]
        elif index == 1:
            self.data['moth_mask'] = data_to_add
        elif index == 2:
            self.data['forewing_hinge_points'] = data_to_add
            
    def run(self):
        while self.exit_code == ExitCode.READY:
            self.current.run()

            # Move to the next step based on the exit code
            if self.current.exit_code == ExitCode.PREV:
                self.prev()
            elif self.current.exit_code == ExitCode.DONE:
                self.update_data(self.index)
                self.backup_step(self.index)
                self.next()
                self.clear(self.index)
            elif self.current.exit_code == ExitCode.NEXT:
                self.restore_step(self.index)
                self.next()
