# Robot remote control with VR
项目地址：https://github.com/Zhuochenn/teleUR?tab=readme-ov-file
拉取branch版本：tactile，而不是main
1. Realsense相机配置：相机连接电脑，运行realsense_show.py，查看端口，然后运行markerless.py查看各个端口的相机屏幕
2. 找视触觉相机的端口：运行`cam_port.py`检测可用相机端口，运行`markerless.py`，指定端口查看画面
3. UR5启动：按控制板顶部开机按钮，点左下角开启，然后点start。如果已经安装平行夹爪，点右上角UR+，打开手。
4. 网络连接：
   1. robot ip address设置为：10.40.101.10（检查是否被修改），指定机器人的host ip（主机）: 10.40.101.1
   2. 笔记本插网线，
      1. 设置ip：`sudo ip addr add 10.40.101.1/24 dev enp132s0`
      2. 测试连接：`ping 10.40.101.10`，如果ping不通，可以插拔网线
   3. 机器人面板右上角选择remote模式
5. VR眼镜按左边圆点开机
6. 加载节点：运行`python launch_nodes.py`，不要关掉
7. 眼镜操作设置：
   1. 戴眼镜，点开主页的`notification - usb detected`，点击允许debugging，在主页选择底部状态栏的安卓图标，点击弹出窗口，进入
   2. 在屏幕按食指，感受到震动说明已连接
   3. 把眼镜倒放在桌上，手柄放在眼镜前面让它检测到，按下食指确认依然有震动
8. 开启机械臂控制：查看`QUICK_START.md`
   1. 例如：在不使用tactile且不保存数据时，直接运行机械臂控制：`python run_env.py --no-use-tactile`，如果看到下面的结果，说明VR眼镜没有获得权限，可以拔插一下眼镜重新设置。
```commandline
Device is visible but could not be accessed.
Run `adb devices` to verify that the device is visible and accessible. 
```
如果卡住，显示下面内容，可能是眼镜连接问题，重新连接后跑代码
```
Quest agent created
(1, 480, 640) uint16
```
9. 手柄控制：按住食指的时候移动手柄，机械臂会跟着动，松开时解绑
10. 机器人关机：按面板右上角shut down，注意机器人的手远离桌面，因为会抖一下
11. 眼镜关机：长按左边的圆点按钮，直到出现一个power off图标

