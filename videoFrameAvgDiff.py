#!/usr/bin/python 

# videoFrameAvgDiff.py - Calculates average pixel difference between each consecutive frame in a video to detect stagnant periods
# Mainained by Anthony Spears - aspears@gatech.edu
# Created: 10 Nov 2020
# Internal note: Using virtualenvwrapper - $ workon frame_grabber 

import ffmpeg  
import cv2
import sys
import csv
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Recieved ", len(sys.argv), " arguments")
        print("Usage: python OculusFrameSync.py /home/user/video.avi")
        return
    in_filename=str(sys.argv[1])
    print("Opening video file: ", in_filename)
    csv_outname="outputDiff.csv"
    csvout=open(csv_outname,"w")
    csvout.write("Frame1,Frame2,AvgPixDiff\n")
    
    # Open video file for reading
    in_vid = cv2.VideoCapture(in_filename)

    #Exit if video not opened.
    if not in_vid.isOpened():
        print('Can\'t open input video file')
        sys.exit()

    # Read first image and get resolution
    ok, frame = in_vid.read()
    if not ok:
        print('Can\'t read video file')
        sys.exit()
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    #cv2.imwrite("frame1.png",frame)

    #mask = cv2.imread('/media/aspears3/Data/oculus_template_cleaned.png')
    #mask_flipped = cv2.flip(mask,0)
     
    #frame_masked=cv2.bitwise_and(frame,frame,mask)

    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    prev_frame=frame_gray.copy()

    # Get frame rate and fourcc of input video.
    fps = in_vid.get(cv2.CAP_PROP_FPS)
    print("input video fps:", fps)
#    fourcc = in_vid.get(cv2.CAP_PROP_FOURCC)
#    print("input video fourcc: ",fourcc)
    framecount = in_vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print("input video framecount:", framecount) 

    #    in_vid.set(cv2.CAP_PROP_POS_FRAMES,framecount-20)

    frame_counter = 1
    frame_counter_prev = 1

    while(in_vid.isOpened()):
        # Read a new frame
        ok, frame = in_vid.read()
        if not ok:
            break
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('video',frame)
        #cv2.waitKey(0)

        frame_counter+=1
        print(frame_counter)
        print("input video framecount:", framecount) 
        #if(frame_counter==3.0):
        #    print("Wrote last frame")
        #    cv2.imwrite("frame3.png",frame)

        # Calculate average pixel diff
        diff=cv2.absdiff(frame_gray,prev_frame)
        avgPixDiff=cv2.mean(diff)
        avgPixDiff0=avgPixDiff[0]
        print("Mean pixel shift: ", avgPixDiff0)
        
        csvout.write(str(frame_counter_prev) + "," + str(frame_counter) + "," + str(avgPixDiff0) + "\n")
        prev_frame=frame_gray.copy()
        frame_counter_prev=frame_counter

    csvout.close()

if __name__ == "__main__":
    main()
