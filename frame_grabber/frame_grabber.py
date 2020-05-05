import ffmpeg  # Note: import ffmpeg is not a must, when using OpenCV solution.
import cv2
import sys

out_filename = 'out.mp4'

# Build synthetic video and read binary data into memory (for testing):
#########################################################################
in_filename = '/media/aspears3/Data/test.mov' #'in.mp4'
width, height = 320, 240
fps = 1  # 1Hz (just for testing)

# Build synthetic video, for testing:
# ffmpeg -y -f lavfi -i testsrc=size=320x240:rate=1 -c:v libx264 -crf 18 -t 50 in.mp4
(
    ffmpeg
    .input('testsrc=size={}x{}:rate={}'.format(width, height, fps), f='lavfi')
    .output(in_filename, vcodec='libx264', crf=18, t=50)
    .overwrite_output()
    .run()
)
#########################################################################

# Open video file for reading
in_vid = cv2.VideoCapture(in_filename)

#Exit if video not opened.
if not in_vid.isOpened():
    print('Cannot open input video file')
    sys.exit()

# Read first image (for getting resolution).
ok, frame = in_vid.read()
if not ok:
    print('Cannot read video file')
    sys.exit()
cv2.imwrite("frame%d.png" % 0, frame)

width, height = frame.shape[1], frame.shape[0]

# Get frame rate of input video.
fps = in_vid.get(cv2.CAP_PROP_FPS)

# Create video writer
# 264 doesn't come by default with the default installation of OpenCV, but I preferred using H.264 (supposed to be better than XVID).
# https://stackoverflow.com/questions/41972503/could-not-open-codec-libopenh264-unspecified-error
# (I had to download openh264-1.8.0-win64.dll)
out_vid = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))


frame_counter = 0

while True:
    # Write every 10th frame to output video file.
    if ((frame_counter % 10) == 0):
        out_vid.write(frame)
        cv2.imwrite("frame%d.png" % frame_counter, frame)

    # Read a new frame
    ok, frame = in_vid.read()
    if not ok:
        break

    frame_counter += 1
    out_vid.release()





# Selecting one every n frames from a video using FFmpeg:
# https://superuser.com/questions/1274661/selecting-one-every-n-frames-from-a-video-using-ffmpeg 
# ffmpeg -y -r 10 -i in.mp4 -vf "select=not(mod(n\,10))" -vsync vfr -vcodec libx264 -crf 18  1_every_10.mp4

out_filename = '1_every_10.mp4'

# Important: set input frame rate to fps*10
(
    ffmpeg
    .input(in_filename, r=str(fps*10))
    .output(out_filename, vf='select=not(mod(n\,10))', vsync='vfr', vcodec='libx264', crf='18')
    .overwrite_output()
    .run()
)
