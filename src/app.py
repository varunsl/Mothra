import argparse
import os
import sys

import numpy as np
import cv2

from GUI import FullGUI

def main(input_folder, output_folder, reviewing):    
    extensions = ['.jpg', '.JPG', '.png', '.PNG']
    for file in os.listdir(input_folder):
        if any(file.endswith(ext) for ext in extensions):                                        
            file_path = str(os.path.join(input_folder, file))
            img = cv2.imread(file_path)
            img = cv2.resize(img, (0,0), fx=0.15, fy=0.15)
                        
            mGUI = FullGUI.FullGUI(img)
            mGUI.run()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input folder")
    ap.add_argument("-o", "--output", required=True, help="path of the output folder")
    ap.add_argument("-r", "--review", dest='review', action='store_true',
        help="review previously digitized images")
    
    args = ap.parse_args()
    # handle ini folder
    main(args.input, args.output, args.review)