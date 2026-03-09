import pickle

file="shared/data/bc_data/0224_193531/2026-02-24T19-35-31-568492.pkl"
with open(file, "rb") as f:
    data=pickle.load(f)  # <class 'dict'>, len=14
    # dict_keys(['base_camera_rgb', 'base_camera_depth', 'tactile_left_rgb', 'tactile_left_depth', 'tactile_right_rgb', 'tactile_right_depth', 'joint_positions', 'joint_velocities', 'eef_speed', 'ee_pos_quat', 'gripper_position', 'touch', 'activated', 'control'])
    print(len(data['base_camera_rgb']))