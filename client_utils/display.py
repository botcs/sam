import cv2
import numpy as np

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.width()
    h = rect.height()

def drawBBox(bgrImg, bb, args, id_counter=None, consecutive_occurrence=None, CARD2NAME=None):
    '''
        Draws a bounding box around a single face, shows stats if given
        
        Args: 
            bgrImg - [H, W, C] shaped numpy array, where C=3 and order is B G R
            bb - [xmin, ymin, w, h] of the bounding box, type: dlib.rectangle
            
            id_counter - {ID:occurrence}  (optional)
            consecutive_occurrence - counter for gradual color shift (optional)
            
        Returns:
            bgrImg - the modified image
              
    '''
    # This is handled on server side
    #x, y, w, h = rect_to_bb(bb)
    x, y, w, h = bb
    
    x_offset = 80
    y_offset = 40
    radius_addition = 15
    font_scale = 1.5

    # if the drawn bbox is the main bbox
    if id_counter is not None:
        percentage = id_counter[0][1]/args.k*100
        thickness = 2    
        if percentage < args.threshold or id_counter[0][0].find('>') > -1:
            color = (0, 0, 200)
            text = '<UNK>'

        else:
            # gradually turning from white to green the 
            ratio = max(args.consecutive - consecutive_occurrence, 0) / args.consecutive
            color = (ratio * 200, 200, ratio * 200)
            shown_ID = id_counter[0][0]
            if CARD2NAME.get(shown_ID) is not None:
                shown_ID = CARD2NAME[id_counter[0][0]]
            text = '%s'%(shown_ID)
        
        # Show statistics below the bounding box
        cv2.putText(bgrImg, text, (x + x_offset-w//2, y + h + y_offset + radius_addition),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, thickness, cv2.LINE_AA)
    
        circle_color = color
        circle_thickness = 3 
        if consecutive_occurrence >= args.consecutive:
            circle_color = (0, 230, 0) 
            circle_thickness = 5
            
    # if the drawn bbox is just in the background
    else:
        circle_color = (220, 220, 220)
        circle_thickness = 1 
    
    # Finally drawing the circle around the face
    cv2.circle(bgrImg, (x+w//2, y+h//2), w//2+radius_addition, circle_color, circle_thickness)
    return bgrImg        



bannerDict = {}
def addbanner(img, bannerPath, concat=True):
    '''
        Show an external image on the live display
        Args:
            img - [H, W, C] input image, opencv format
            bannerPath - location of the external image
            concat - replace the top segment of the img if False
        
        Returns:
            img - the modified image
            
    '''

    global bannerDict
    banner = bannerDict.get(bannerPath)
    if banner is None:
        banner = cv2.imread(bannerPath)
        bannerDict[bannerPath] = banner
        
    if concat:
        img = np.concatenate([banner, img])
    else:
        img[:banner.shape[0]] = banner
    
    return img
    
    


def drawBanner(img, id_counter=None, CARD2NAME=None, AUTHORIZED_ID=None, current_timeout=0):
    '''
        Shows live statistics of the recognition algorithm
    '''

    H, W, C = img.shape
    H = 50
    
        
    if AUTHORIZED_ID is None:
        img = addbanner(img, 'passive-banner.png')
    else:
        img = addbanner(img, 'active-banner.png')
    
    if current_timeout > 1:
        text = 'Connecting to server... (%3d)'%current_timeout
        cv2.putText(img, text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0,0,0), 
                    thickness=1, lineType=cv2.LINE_AA)
        return img


    if id_counter is not None:    
        for i, (n, c) in enumerate(id_counter[:3]):
            registered = False
            if CARD2NAME.get(n) is not None:
                registered = True
                n = CARD2NAME.get(n)
            text = '%s (%2d)'%(n, c)
            
            # If user is registered and ID is the first then use Green text
            if i == 0:
                if registered:
                    color = (40, 200, 40)
                else:
                    color = (0, 0, 200)
            else:
                color = (0, 0, 0)
            
            cv2.putText(img, text, (30 + i*150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, 
                thickness=1, lineType=cv2.LINE_AA)

    else:
        text = 'No IDs to show...'
        cv2.putText(img, text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0,0,0), 
                    thickness=1, lineType=cv2.LINE_AA)
                
    return img

