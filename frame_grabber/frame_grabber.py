# frame_grabber.py - Frame grabber for full resolution stills from video
# Mainained by Anthony Spears aspears@gatech.edu

import ffmpeg  
import cv2
import sys

#########################################################################
in_filename = '/media/aspears3/Data/test.mov'
fps_out = 5
#########################################################################

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
cv2.imwrite("frame%d.png" % 0, frame)

width, height = frame.shape[1], frame.shape[0]

# Get frame rate of input video.
fps = in_vid.get(cv2.CAP_PROP_FPS)
print("fps:", fps)

frame_counter = 0

while True:
    if ((frame_counter % (fps/fps_out)) == 0):
        cv2.imwrite("frame%d.png" % frame_counter, frame)

    # Read a new frame
    ok, frame = in_vid.read()
    if not ok:
        break

    frame_counter += 1

# Selecting one every n frames from a video using FFmpeg:
# https://superuser.com/questions/1274661/selecting-one-every-n-frames-from-a-video-using-ffmpeg 
# ffmpeg -y -r 10 -i in.mp4 -vf "select=not(mod(n\,10))" -vsync vfr -vcodec libx264 -crf 18  1_every_10.mp4

##(
##    ffmpeg
##    .input(in_filename, r=str(fps*10))
##    .output(out_filename, vf='select=not(mod(n\,10))', vsync='vfr', vcodec='libx264', crf='18')
##    .overwrite_output()
##    .run()
##)
