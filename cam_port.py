import cv2
import os
import time

def list_all_video_devices():
    devices = []
    for name in os.listdir('/dev'):
        if name.startswith('video') and name[5:].isdigit():
            devices.append(f'/dev/{name}')
    return sorted(devices, key=lambda x:int(x.split('video')[1]))

def test_device(path):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    time.sleep(0.2)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        return ret
    cap.release()
    return False

if __name__ == "__main__":
    all_devices = list_all_video_devices()
    print("Found video devices:", all_devices)
    working_devices = []
    for dev in all_devices:
        if test_device(dev):
            print(f"{dev} is available and working.")
            working_devices.append(dev)
        else:
            print(f"{dev} cannot be opened or no frame.")
    print("Usable camera devices:", working_devices)