# 1. 采集数据
python run_env.py --save-data --tactile-left-camera-id 22 --tactile-right-camera-id 24

# 2. 查看数据（会弹出交互窗口）
python visualize_data.py ./shared/data/bc_data/$(ls -t ./shared/data/bc_data/ | head -1)

# 3. 分析数据（查看统计信息）
python analyze_data.py ./shared/data/bc_data/$(ls -t ./shared/data/bc_data/ | head -1)

# 4. 如果满意，可以导出视频
python visualize_data.py ./shared/data/bc_data/YOUR_FOLDER --export-video demo.mp4 --fps 10
python visualize_data.py ./shared/data/bc_data/0224_184609 --export-video demo_test.mp4 --fps 10