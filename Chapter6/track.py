# ----------------------------Modules----------------------------
from facenet_pytorch import MTCNN 
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw


# --------------------------MTCNN Model--------------------------
mtcnn = MTCNN(keep_all=True, device='cpu')


# --------------Loading a Video for Face Recognition-------------
video = mmcv.VideoReader('./data/video.mp4') 
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]


## we will track each frame using 
frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')

    # Detect faces within each boxes
    boxes, _ = mtcnn.detect(frame)
    
    # Draw a rectagle around the faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    
    # Add to frame list
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
print('\nDone')


# ---------------------Saving Tracked Video----------------------
dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
video_tracked = cv2.VideoWriter('./results/video_tracked.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()