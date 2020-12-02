import math
import cv2
import numpy as np

GRADIENT_IMG=True #Create synthetic gradient image
BEARING_GRADIENT=False #Create Bearing Gradient Synthetic Img Instead of Range Grandient
POLAR_GRID=False #Create grid in original polar image
CARTESIAN_GRID=True #Create grid in converted cartesian image

#img = cv2.imread('gradient-reversed.jpg',0)

#Create synthetic img
fov=130 #sonar fov degs - either 130 or 70 for m750d
nRanges=500
nBearings=256
height=nRanges
width=nBearings
img = np.zeros((height,width,3), np.uint8)

cv2.imshow('img',img)
cv2.waitKey(0)

rows,cols,depth = img.shape

###################################
#Create gradient image
if GRADIENT_IMG:
    for i in range(height):
        for j in range(width):
            if BEARING_GRADIENT:
                img[i,j]=[j*(256/width),j*(256/width),j*(256/width)]
            else:
                img[i,j]=[i*(256/height),i*(256/height),i*(256/height)]
###################################
###################################
#Create grid in polar image (not preferred since it is then stretched in cart coords)
#First create bearing lines
if POLAR_GRID:
    for i in range(9):
        if i==0:
            cv2.line(img,(0,0),(0,rows),(255,255,255),1)
        else:
            cv2.line(img,(i*32-1,0),(i*32-1,rows),(255,255,255),1)
    #Now create range lines
    for i in range(6):
        if i==0:
            cv2.line(img,(0,0),(cols,0),(255,255,255),1)
        else:
            cv2.line(img,(0,i*100-1),(cols,i*100-1),(255,255,255),1)
###################################

cv2.imshow('img',img)
cv2.waitKey(0)

#now rotate image to match opencv polar coordinate frame
rotated=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rows,cols,depth = rotated.shape
cv2.imshow('img',rotated)
cv2.waitKey(0)

#Add padding to polar image to show full 360 degrees for future polar2cart conversion
padding_pix=round(((nBearings/fov)*360 - nBearings)/2)
padded=cv2.copyMakeBorder(rotated,padding_pix,padding_pix,0,0,cv2.BORDER_CONSTANT)
rows,cols,depth = padded.shape

cv2.imshow('img',padded)
cv2.waitKey(0)

dst_size = (nRanges, nRanges*2)
dst_center = (nRanges, nRanges)
cart = cv2.warpPolar(padded, dst_size, dst_center, 500, flags=(cv2.WARP_INVERSE_MAP))
rows,cols,depth = cart.shape

#now rotate image to match sonar/vehicle coordinate frame
cart_rotated=cv2.rotate(cart, cv2.ROTATE_90_CLOCKWISE)
rows,cols,depth = cart_rotated.shape
cv2.imshow('img',cart_rotated)
cv2.waitKey(0)

###################################
#Create grid in cartesian image
if CARTESIAN_GRID:
    max_radius=nRanges
    startAngle=-25
    endAngle=-155
    startAngleRad=startAngle*math.pi/180
    endAngleRad=endAngle*math.pi/180
    thickness=1

    #First create bearing lines
    for i in range(5):
        radius=round((i+1)*max_radius/5)
#        cv2.ellipse(dst, dst_center, axes, angle, startAngle, endAngle, (255,255,255), thickness)
        cv2.ellipse(cart_rotated, dst_center, (radius, radius), 0, startAngle, endAngle, (255,255,255), 1)
        
    #Now create range lines
    for i in range(5):
        angle=startAngleRad-i*(fov*math.pi/180)/4
        end_point=(dst_center[0]+round(max_radius*math.cos(angle)), dst_center[1]+round(max_radius*math.sin(angle)))
        cv2.line(cart_rotated,dst_center,end_point,(255,255,255),1)
###################################
cv2.imwrite("cartesianSonarOut.png", cart_rotated)
print("DONE")
cv2.imshow('img',cart_rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
