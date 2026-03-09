import numpy as np
from utils import find_marker,find_marker_centers, refresh_dir, plot_marker_delta
import cv2
import argparse
import os
from functools import partial

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--cam_id", type=int, default=4)
    io_parser.add_argument("--output_dir",type=str,default="marker_shift")
    lk_parser = parser.add_argument_group()
    lk_parser.add_argument("--winSize",type=int, nargs=2, default=[15,15])
    lk_parser.add_argument("--maxLevel",type=int, default=2)
    marker_parser = parser.add_argument_group()
    marker_parser.add_argument("--morphop_size",type=int,default=5)
    marker_parser.add_argument("--morphop_iter",type=int,default=1)
    marker_parser.add_argument("--morphclose_size",type=int,default=5)
    marker_parser.add_argument("--morphclose_iter",type=int,default=1)
    marker_parser.add_argument("--dilate_size",type=int,default=3)
    marker_parser.add_argument("--dilate_iter",type=int,default=0)
    marker_parser.add_argument("--marker_range",type=int,nargs=2,default=[145,255])
    marker_parser.add_argument("--value_threshold",type=int,default=90)
    return parser.parse_args()


if __name__ == "__main__":
    args = options()
    calib_find_marker = partial(find_marker,
        morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morphop_size, args.morphop_size)),
        morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morphclose_size, args.morphclose_size)),
        dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate_size, args.dilate_size)),
        mask_range=args.marker_range,
        min_value=args.value_threshold,
        morphop_iter=args.morphop_iter,
        morphclose_iter=args.morphclose_iter,
        dilate_iter=args.dilate_iter
    )
    refresh_dir(args.output_dir)
    stream = cv2.VideoCapture(args.cam_id)
    track_func = partial(cv2.calcOpticalFlowPyrLK, winSize=args.winSize, maxLevel=args.maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    img_cnt = 0
    ret, ref_img = stream.read()
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    if ret == False:
        raise RuntimeError('camera abnormal.')
    ref_marker = calib_find_marker(ref_img)
    p0 = np.array(find_marker_centers(ref_marker),dtype=np.float32)
    # np.savetxt(args.ref_marker_center, p0, fmt='%.4f')
    while True:
        ret, track_img = stream.read()
        if ret == False:
            break
        track_gray = cv2.cvtColor(track_img, cv2.COLOR_RGB2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(ref_gray, track_gray, p0, None)
        st = st.reshape(-1)
        kpt0 = p0[st == 1]
        kpt1 = p1[st == 1]
        res = np.hstack((kpt0, kpt1-kpt0))
        np.savetxt(os.path.join(args.output_dir, "%04d.txt"%img_cnt), res, fmt="%.4f")
        img_cnt += 1
        plot_img = plot_marker_delta(track_img, kpt1, kpt1-kpt0, scale=6,arrow_color=(0,0,255))
        cv2.imshow('demo',plot_img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord('s'):
            num = len(os.listdir("images/"))
            cv2.imwrite(f"images/{num:04d}.png",plot_img)
            print(f"image saved!:{num:04d}.png")

    cv2.destroyAllWindows()
    # stream.close()