import pickle
import matplotlib.pyplot as plt

pkl_file = "shared/data/bc_data/0606_113104/2025-06-06T11-31-04-942786.pkl"

# Load the pickle file
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# Prepare RGB and depth images
rgb = data['base_camera_rgb'].squeeze()    # shape: (480, 640, 3)
depth = data['base_camera_depth'].squeeze()# shape: (480, 640)

# Create a single figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Show RGB image
axs[0].imshow(rgb)
axs[0].set_title("Base Camera RGB")
axs[0].axis('off')

# Show depth image
im = axs[1].imshow(depth, cmap='inferno')
axs[1].set_title("Base Camera Depth")
axs[1].axis('off')
fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label='Depth')

plt.tight_layout()
plt.show()

# Print other information
print("Joint Positions:", data['joint_positions'])
print("Joint Velocities:", data['joint_velocities'])
print("EEF Speed:", data['eef_speed'])
print("EE Pos/Quat:", data['ee_pos_quat'])
print("Gripper Position:", data['gripper_position'])
print("Touch Sensors:", data['touch'])
print("Activated:", data['activated'])
print("Control:", data['control'])