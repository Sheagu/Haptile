# Robot remote control with Meta Quest 3
Robot remote control is based on the tactile branch of [teleUR](https://github.com/Zhuochenn/teleUR) project.

Hardware preparation
- Connect the RealSense and tactile cameras to the control computer.
- Power on the UR5e robot and enable the gripper.
- Configure the robot and host network interfaces on the same subnet, then switch the robot to Remote Control mode.
- Power on the Meta Quest headset.
- Start the hardware nodes and keep the process running: `python launch_nodes.py`.

Meta Quest headset setup
- Put on the headset and allow USB debugging when prompted.
- Point the controller at the black application window and press the right index trigger. A vibration indicates that the controller is connected.
- Place the headset upside down on the table while keeping the controller within its tracking area.
- Press the right index trigger again and confirm that the controller still vibrates. This verifies that the connection and tracking remain active.

Robot control
- Start a trajectory: Press the right joystick (RJ) once. The robot moves to its initial pose and starts recording.
- Control the robot: Hold the right index trigger (rightTrig) and move/rotate the controller. Release the trigger to pause robot motion. Always release the index trigger before repositioning your hand or controller. This prevents unintended robot motion when control is activated again.
- Control the gripper: Hold the side grip button (rightGrip) to increase the gripper command. Hold the A button to decrease it.
- Reset tactile tracking: Press B to use the current tactile images as new marker-tracking reference frames.
- Save the trajectory: Double-click the right joystick within approximately 0.5 seconds.
- Discard the trajectory: Triple-click the right joystick within approximately 0.5 seconds.

# Data storage
Collected data is stored in `shared/data/bc_data` by default.