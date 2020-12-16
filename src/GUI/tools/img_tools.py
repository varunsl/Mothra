import numpy as np
import cv2

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def hough(img):
    '''
    Detects lines in an image.

    Args:
        img: The input image
    Returns:
        A list of the endpoints of the detected lines
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(gray)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    endpoints = []
    for rho, theta in lines[0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        endpoints.append((pt1, pt2))
    return endpoints

def object_detector(img):
    '''
    Crude object detection which expects a plain background.

    Args:
        img: The input image
    Returns:
        A list of unsorted tuples (x, y, width, height) corresponding to
        bounding boxes of the objects detected
    '''
    # We first apply Canny edge detection to find where objects are
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray, 100, 200)
    edges = auto_canny(gray, sigma=0.75)

    # Then apply morphological dilation so the edges are thicker
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # The objects should basically be blobs now, so we find contours
    cntrs, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(cntr) for cntr in cntrs]

def sliding_window(image, stepSize, windowSize):
    '''
    Yields pieces of an image given by a sliding window.

    Args:
        img: The input image
        step_size: An integer controling how much the window moves
        window_size:An (x, y) tuple controlling the window size

    Returns:
        Portions of the image inside the sliding window
    '''
    height, width = image.shape[:2]
    window_width, window_height = windowSize
    
    for y in range(0, height, stepSize):
        for x in range(0, width, stepSize):
            yield (x, y, image[y:y + window_height, x:x + window_width])

def vertical_scan(image, stepSize):
    '''
    Yields horizontal slices of an image going from bottom to the top.

    Args:
        img: The input image
        stepSize: The height of the slices

    Returns:
        Horizontal slices of the image
    '''

    height, width = image.shape[:2]

    for y in range(height, stepSize, -1 * stepSize):
        yield (y, image[y - stepSize:y, :])

def remove_color(image, coords, tolerance=(10, 10, 40)):
    '''
    Subtracts an input pixel's color from an image.

    Args:
        image: The input image in HSV format
        coords: A tuple representing the location of the pixel
        tolerance: A tuple representing the HSV range to use

    Returns:
        A copy of the original image with the color removed
    '''
    x, y = coords
    pixel = image[y, x]
    
    upper =  np.array(pixel) + np.array(tolerance)
    lower =  np.array(pixel) - np.array(tolerance)

    mask = cv2.inRange(image, lower, upper)
    mask = np.invert(mask)
    
    color_removed = cv2.bitwise_and(image, image, mask=mask)
    return color_removed

def flood_fill(img, seed, color=(0, 0, 0), lo=20, hi=20):
    '''
    Fills a connected component in an image with the given color.

    Args:
        img: The input image
        seed: A point in the connected component
        color: The color to fill the image with
        lo: Parameter controlling the tolerance of the floodfill
        hi: Another parameter controlling tolerance
    Returns:
        A copy of the image with the region flooded
    '''
    flooded = img.copy()
    height, width = flooded.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)

    flags = 4 | cv2.FLOODFILL_FIXED_RANGE

    cv2.floodFill(flooded, mask, seed, color, (lo,)*3, (hi,)*3, flags)
    return flooded

def flood_remove(img):
    result = img.copy()

    height, width = result.shape[:2]
    step = min(height // 5, width // 5)

    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(greyscale, 100, 200)

    # The background should be where there are no edges, so we use a
    # sliding window to remove areas corresponding to empty spots in
    # the result of the canny edge detection
    for (x, y, window) in sliding_window(edges, step, (step, step)):
        if cv2.mean(window) == (0.0, 0.0, 0.0, 0.0):
            result = flood_fill(result, (x, y))
    return result
    

def remove_background(img, img_hsv):
    '''
    Uses color masking to remove the (plain) background of an image.

    Args:
        img: The input image
        img_hsv: The input image in HSV format

    Returns:
        The image with the background removed in HSV format
    '''
    result = img_hsv

    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(greyscale, 100, 200)

    # The background should be where there are no edges, so we use a
    # sliding window to remove colors corresponding to empty spots in
    # the result of the canny edge detection
    for (x, y, window) in sliding_window(res, step_size=128, windowSize=(64,64)):
        if cv2.mean(window) == (0.0, 0.0, 0.0, 0.0):
            pixel = img_hsv[y, x]
            remove_color(result, pixel)
    return result

def grab_cut(img, rect, mask=None, num_iterations=5):    
    height, width, _ = img.shape

    mode = cv2.GC_INIT_WITH_MASK

    if mask is None:
        mode = cv2.GC_INIT_WITH_RECT
        mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, num_iterations, mode)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img, mask
    
def mask_from_contour(gray_img, cnts=None):
    '''
    Creates a filled in black and white mask from contours.

    Args:
        gray_img: The input image
        cnt: The input contour list (optional)

    Returns:
        A black and white mask with the contours filled.
    '''
    if cnts is None:
        cnts = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    new_mask = np.zeros_like(gray_img)
    new_mask = np.dstack((new_mask, new_mask, new_mask))
    
    for cnt in cnts:
        cv2.drawContours(new_mask, [cnt], 0, (255, 255, 255), -1)

    new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
    return new_mask

def find_contour(gray_img, approx=True):
    blur = cv2.medianBlur(gray_img, 5)
    if approx:
        cnts = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    else:
        cnts = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return max(cnts, key = cv2.contourArea)

def centroid(cnt):
    '''
    Calculate the centroid of a contour.

    Args:
        cnt: A contour calculated by cv2.findCountours

    Returns:
        The centroid.
    '''
    moments = cv2.moments(cnt)
    return (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

def convexity_defects(gray_img):
    cnt = find_contour(gray_img)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    candidate_points = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(cnt[f][0])
        candidate_points.append((far, d))
    s = sorted(candidate_points, key=lambda elem : -elem[1])
    return [pt for pt, d in s]

def convexity_defects_1(gray_img):
    cnt = find_contour(gray_img)
    canvas = np.zeros_like(gray_img)
    canvas = np.dstack((canvas, canvas, canvas))
    cv2.drawContours(canvas, [cnt], 0, (255,0,0), 3)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(canvas,start,end,[0,255,0],2)
        cv2.circle(canvas,far,3,[0,0,255],-1)
    show(canvas)
    return canvas
