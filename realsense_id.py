import pyrealsense2 as rs

def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return
    for dev in devices:
        print(f"Device Name: {dev.get_info(rs.camera_info.name)}")
        print(f"Serial Number (ID): {dev.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware Version: {dev.get_info(rs.camera_info.firmware_version)}")
        print('-' * 40)

if __name__ == '__main__':
    list_realsense_devices()