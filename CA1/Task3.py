import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os


FILE_NAME = "Original_Vid.mp4"
input_file_path = os.path.join(".", FILE_NAME)


if __name__ == "__main__":


    # Part 1
    vid = cv2.VideoCapture(input_file_path)

    if not vid.isOpened():
        print("Error: Could not open video.")
        exit()

    print(vid)
    print(type(vid))



    # Part 2
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = vid.get(cv2.CAP_PROP_FPS)

    duration = frame_count / fps

    print("Frame count:", frame_count)
    print("Fps:", fps)
    print(f"Duration: {duration:.3f}")



    # Part 3
    out_file_path = "output_saltpepper_vid.mp4"
    amount   = 0.1   # fraction of pixels to corrupt per frame (10%)
    s_vs_p   = 0.5    # salt vs pepper ratio (0..1) 0.5 = equal

    w  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))   

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_file_path, fourcc, fps, (w, h))

    p_salt   = amount * s_vs_p
    p_pepper = amount * (1 - s_vs_p) 


    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # random mask per frame
        r = np.random.rand(h, w)

        noisy = frame.copy()
        # pepper -> set to 0 (black)
        pepper_mask = r < p_pepper
        noisy[pepper_mask] = 0

        # salt -> set to 255 (white)
        salt_mask = r > (1 - p_salt)
        noisy[salt_mask] = 255

        out.write(noisy)

    vid.release()
    out.release()
    print(f"Saved: {out_file_path}")
