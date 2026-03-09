import time
import cv2
import numpy as np
import os
import json

def open_camera():
    # --- 1. 设置摄像头索引 ---
    camera_index = 4  # windows的是0，linux的是2,4

    # 注意：在 Windows 上使用 cv2.CAP_DSHOW 后端通常能更稳定地控制曝光
    cap = cv2.VideoCapture(camera_index)# , cv2.CAP_DSHOW)

    # 检查是否成功打开
    if not cap.isOpened():
        print(f"无法打开摄像头 (Index {camera_index})。请尝试修改 camera_index 为 0。")
        return

    # --- 2. 设置分辨率与帧率 (可选) ---
    # xuyang建议用这个分辨率（30万像素）就够了
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)  #

    # --- 3. (关键) 锁定曝光设置 ---
    # 做触觉时，必须关闭自动曝光！
    # -1 到 -10 是常见的曝光值，数值越小画面越暗。
    # 如果画面全黑，请把这个值调大（例如 -3）；如果太亮，调小（例如 -7）。
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 通常代表“手动模式”
    cap.set(cv2.CAP_PROP_EXPOSURE, -5.5)  # 手动曝光值 (根据实际情况调整)

    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 计算缩放比例 (让图像适配屏幕，假设屏幕高度约 1080)
    max_display_height = 1440  # 显示窗口的最大高度
    scale = max_display_height / actual_height if actual_height > max_display_height else 1.0
    display_width = int(actual_width * scale)
    display_height = int(actual_height * scale)

    print("摄像头已启动。按 'q' 键退出，按 's' 键保存截图。")

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧 (stream end?). Exiting ...")
            break
        # --- 4. 图像处理 (模拟触觉算法输入) ---
        # 保存时用原始 frame，显示时用缩放后的 display_frame
        display_frame = cv2.resize(frame, (display_width, display_height),
                                   interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Raw Camera (Color)', display_frame)
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 q 退出
            break
        elif key == ord('s'):  # 按 s 保存图片用于写论文/调试
            filename = f"tactile_sample_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"已保存截图: {filename}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def show_point_position(img_path):
    '''显示被选的点的坐标
    '''
    img = cv2.imread(img_path)  # todo 读视频第一帧
    if img is None:
        print("错误：无法读取图像")
        return None

    # 创建一个副本，避免在原图上涂鸦破坏数据
    display_img = img.copy()
    points = []
    win_name = "Select Point"

    # 构造要传递给回调函数的“包裹”
    data_package = {
        'image': display_img,
        'points': points,
        'window': win_name
    }

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, select_points, data_package)

    print("请点击任意位置，查看坐标")
    cv2.imshow(win_name, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_and_warp(frame, selected_points, out_width=1600, out_height=1100):
    """
    将不规则四边形区域映射为矩形
    :param frame: 原始视频帧
    :param selected_points: 你鼠标点选的 4 个点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    :param out_width: 输出矩形的宽度
    :param out_height: 输出矩形的高度
    """
    # 1. 整理源点坐标 (必须是 float32 类型)
    # 确保顺序与目标点对应：通常为 [左上, 右上, 右下, 左下]
    src_pts = np.float32(selected_points)

    # 2. 定义目标点坐标 (即矩形的四个角)
    dst_pts = np.float32([
        [0, 0],                   # 左上
        [out_width, 0],           # 右上
        [out_width, out_height],  # 右下
        [0, out_height]           # 左下
    ])

    # 3. 计算透视变换矩阵 M
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 4. 执行变换
    # dsize 是输出图像的尺寸 (宽, 高)
    warped_img = cv2.warpPerspective(frame, M, (out_width, out_height))
    win_name="rectangle"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)  # 使用 WINDOW_AUTOSIZE，配合 DPI 感知，即可实现 1:1 显示

    while True:
        cv2.imshow(win_name, warped_img)
        # 5. 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 按 s 保存图片用于写论文/调试
            filename = f"rectangle_{int(time.time())}.png"
            cv2.imwrite(filename, warped_img)
            print(f"已保存截图: {filename}")
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

def select_points(event, x, y, flags, param):
    # 解包传入的参数
    image_to_draw = param['image']
    points_list = param['points']
    window_name = param['window']

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,',',y)
        points_list.append((x, y))  # 记录坐标
        cv2.circle(image_to_draw, (x, y), 5, (0, 255, 0), -1)  # 在图像上绘制视觉反馈
        if len(points_list) > 1:
            # 连接上一个点，画出梯形轮廓
            cv2.line(image_to_draw, points_list[-2], points_list[-1], (255, 0, 0), 2)
        if len(points_list) == 4:
            cv2.line(image_to_draw, points_list[3], points_list[0], (255, 0, 0), 2)
        cv2.imshow(window_name, image_to_draw)

def select_sensor_region(img_path):
    '''选择传感器四边形的区域
216 , 163
629 , 115
632 , 373
218 , 344
    '''
    img = cv2.imread(img_path)  # todo 读视频第一帧
    if img is None:
        print("错误：无法读取图像")
        return None

    # 创建一个副本，避免在原图上涂鸦破坏数据
    display_img = img.copy()
    points = []
    win_name = "Select Trapezoid Region"

    # 构造要传递给回调函数的“包裹”
    data_package = {
        'image': display_img,
        'points': points,
        'window': win_name
    }

    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, select_points, data_package)

    print("请按照[左上, 右上, 右下, 左下]的顺序，点击梯形的四个角。")
    cv2.imshow(win_name, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 当 waitKey(0) 结束时，points 列表已经被回调函数填满了
    if len(points) != 4:
        print(f"警告：你点击了 {len(points)} 个点，但我们需要 4 个点。")
    return points

def get_sensor_config(img_path, config_file="sensor_config.json"):
    """获取传感器区域配置，若无则启动交互界面"""
    if os.path.exists(config_file):
        print(f"--- 发现现有配置，正在从 {config_file} 加载 ---")
        with open(config_file, 'r') as f:
            data = json.load(f)
            return np.array(data['points'], dtype="float32")
    else:
        print("--- 未发现配置，请在弹出窗口中选择区域 ---")
        pts = select_sensor_region(img_path)  # 假设它返回一个列表 [(x,y), ...]
        if pts is not None and len(pts) == 4:
            with open(config_file, 'w') as f:
                json.dump({"points": pts}, f)
        else:
            pts=[[0,0],[0,0],[0,0],[0,0]]
        return np.array(pts, dtype="float32")

def get_camera_at_full_res(index, target_w=2560, target_h=1440):
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Windows 建议加上 CAP_DSHOW 标志

    # 1. 尝试设置像素格式为 MJPG (这一步对高分辨率至关重要)
    # MJPG 是压缩格式，能绕过 USB 2.0 的带宽瓶颈
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # 2. 强制设置宽度和高度
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

    # 3. 读取并验证实际生效的分辨率
    actual_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"请求分辨率: {target_w}x{target_h}")
    print(f"实际生效: {actual_w}x{actual_h}")

    return cam

def get_camera_resolution():
    # 使用方法
    cap = get_camera_at_full_res(0)
    ret, frame = cap.read()
    if ret:
        print("最终图像形状:", frame.shape)  # 应该是 (1440, 2560, 3)

def cal_point_distance():
    data=[13 ,32 ,50 ,68 ,84 , 103]
    data_dis=[data[i]-data[i-1] for i in range(1, len(data))]
    print(np.mean(data_dis))

def main():
    # 强制让程序识别显示器的原始物理分辨率，这会禁止 Windows 对 OpenCV 窗口进行 150% 的自动缩放
    # ctypes.windll.shcore.SetProcessDpiAwareness(1)
    img_path = "robo_test/tactile_sample_1773080619.png"
    frame = cv2.imread(img_path)
    if frame is None:
        print("无法读取图片")
        return

    # --- 阶段 1: 获取坐标 (仅在第一次或配置文件被删除时运行) ---
    pts = get_sensor_config(img_path)
    # --- 阶段 2: 映射到矩形 (每次运行都会执行) ---
    tl, tr, br, bl = pts  # 左上, 右上, 右下, 左下
    width = int(np.mean([tr[0], br[0]]) - np.mean([tl[0], bl[0]]))
    height = int(np.mean([br[1], bl[1]]) - np.mean([tl[1], tr[1]]))
    print(f'mapped rectangle: width {width}, height {height}')
    crop_and_warp(frame, pts, out_width=width, out_height=height)


if __name__ == "__main__":
    # open_camera()
    # select_sensor_region("robo_test/tactile_sample_1773080619.png")
    # show_point_position("frame_for_measurement.png")
    # cal_point_distance()
    main()
