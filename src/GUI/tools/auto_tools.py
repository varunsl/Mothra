import sys
import os

import numpy as np
import cv2

import matplotlib.pyplot as plt

from . import img_tools

'''
These functions do the auto-digitizing.
'''

def upside_looper(input_folder):  
    extensions = ['.jpg', '.JPG', '.png', '.PNG']
    for file in os.listdir(input_folder):
        if any(file.endswith(ext) for ext in extensions):                                        
            file_path = str(os.path.join(input_folder, file))
            print(file_path)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (0,0), fx=0.15, fy=0.15)
            print(is_upside_down(img))
            img_tools.show(img)

def is_upside_down(img):
    '''
    Detects whether a moth image is upside down

    Args:
        img: The input image
    Returns:
        True if the image is upside down and false otherwise
    '''
    height, width, _ = img.shape
    
    # We find the longest line in the image, which should correspond
    # to the ruler.

    length = lambda x: abs(x[0][0] - x[1][0])
    
    lines = img_tools.hough(img)
    line = sorted(lines, key=length)[0]

    # We first check that the line is long enough to correspond to the
    # ruler, then we use the heuristic that the ruler should be at the
    # bottom of a correctly oriented image
    return (length(line) >= 0.9 * width) and (line[0][1] < width // 2)

def auto_crop(img):
    '''
    Finds the region that contains a moth in the image

    Args:
        img: The input image
    Returns:
        The opposite corners of the bounding box
    '''
    boxes = img_tools.object_detector(img)

    # Heuristic is that the moth has the second longest bounding box
    x, y, w, h = sorted(boxes, key=lambda b: b[2], reverse=True)[1]
    return (x, y), (x + w, y + h)

def auto_pick(image, image_hsv):
    '''
    Removes the background from a cropped moth image

    Args:
        img: The input image
    Returns:
        f
    '''
    image_hsv = image_hsv.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = img_tools.auto_canny(gray)
    
    for (x, y, window) in img_tools.sliding_window(edges, stepSize=128, windowSize=(64, 64)):
        if cv2.mean(window) == (0.0, 0.0, 0.0, 0.0):
            image_hsv = img_tools.remove_color(image_hsv, (x, y))
    return image_hsv

def auto_bg(image, mask, step=32):
    ret, thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = img_tools.auto_canny(thresh)

    cvs = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    img_tools.show(edges)
    
    for (x, y, window) in img_tools.sliding_window(edges, stepSize=step, windowSize=(step, step)):
        cv2.rectangle(cvs, (x, y), (x + step, y + step), (0, 255, 0), 3)
        if cv2.mean(window) == (0.0, 0.0, 0.0, 0.0):
            cv2.circle(mask, (x + step // 2, y + step // 2), step // 2, 0, -1)
            cv2.circle(cvs, (x + step // 2, y + step // 2), step // 2, (255, 0, 0), -1)
    img_tools.show(cvs)
    return mask

def auto_body(img):
    cnt = img_tools.find_contour(img)
    ctrd = img_tools.centroid(cnt)

    # the original image may have holes in the mask which need to
    # be filled
    mask = img_tools.mask_from_contour(img, [cnt])
    
    raw_left = []
    raw_right = []

    # [(l, r) for l, r in scan_image(mask) if l[0] <= ctrd[0] <= r[0]]
    
    for left_pt, right_pt in scan_image(mask):
        # accept the pieces that the centroid lies inside
        if left_pt[0] <= ctrd[0] <= right_pt[0]:
            raw_left.append(left_pt)
            raw_right.append(right_pt)

    # we apply clustering to remove extraneous points that are part
    # of the wing and not the body
    _, left_pts = cluster(raw_left)
    right_pts, _ = cluster(raw_right)

    # arange the points in the correct order
    height, width = img.shape[:2]

    left_pts = np.vstack(((np.array([[left_pts[0][0], height]])), left_pts))
    left_pts = np.vstack((left_pts, np.array([[left_pts[-1][0], 0]])))

    right_pts = np.vstack(((np.array([[right_pts[0][0], height]])), right_pts))
    right_pts = np.vstack((right_pts, np.array([[right_pts[-1][0], 0]])))
    
    right_pts = np.flip(right_pts, 0)

    body_pts = np.vstack((left_pts, right_pts))

    mask2 = np.zeros_like(mask)
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    cv2.fillPoly(mask2,np.int32([body_pts]),(0,255,255))
    body_mask = cv2.bitwise_and(mask2, mask2, mask=mask)
    img_tools.show(mask2)
    img_tools.show(cv2.addWeighted(body_mask,0.7,cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),0.3,0))
    

def cluster(input_points):
    '''
    Given an array of points, divides them into a left and right cluster based
    on the x coordinate.

    Args:
        input_points: A list of input points. (For example, a list of tuples)

    Returns:
        Two 2D numpy arrays of the computed clusters. The first is the points in
        the left cluster and the second is the right cluster.
    '''
    # convert to format kmeans expects
    pts = np.array(input_points)
    pts = np.float32(pts)

    # select just the x values
    X = pts[:,0]

    # define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(X,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # the left cluster will have the smaller x-value for the center
    left = np.argmin(center.ravel())

    # the right cluster is the other one
    right = 1 - left
    
    # Now separate the data
    L = pts[label.ravel()==left]
    R = pts[label.ravel()==right]
    
    return L, R

def scan_image(img, step_x=2, step_y=4):
    '''Given an input black and white mask, this'''
    height, width = img.shape[:2]

    for y in range(height, step_y, -step_y):
        horizontal_slice = img[y - step_y:y, :]

        left_endpoint = None
        scanning_piece = False

        # scan the window horizontally to look for pieces        
        for x in range(0, width, step_x):
            piece = horizontal_slice[:, x:x + step_x]
            
            if not scanning_piece and np.mean(piece) > 0.1:
                scanning_piece = True
                left_endpoint = (x, y)
            elif scanning_piece and np.mean(piece) < 0.1:
                scanning_piece = False
                yield left_endpoint, (x, y)
    
    
