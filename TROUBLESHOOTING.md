# 故障排除指南

## 🚨 **常见问题与解决方案**

### 1. CUDA驱动问题

#### **问题描述**
```
CUDA error 999: unknown error
RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment
Warp CUDA error 999: unknown error
```

#### **检查步骤**
```bash
# 1. 检查GPU状态
nvidia-smi

# 2. 检查CUDA版本
nvcc --version

# 3. 检查是否有进程占用GPU
ps aux | grep isaac
```

#### **解决方案**
1. **系统重启** (推荐)
   ```bash
   sudo reboot
   ```

2. **重新加载NVIDIA模块**
   ```bash
   sudo rmmod nvidia_uvm
   sudo rmmod nvidia_drm  
   sudo rmmod nvidia_modeset
   sudo rmmod nvidia
   sudo modprobe nvidia
   sudo modprobe nvidia_modeset
   sudo modprobe nvidia_drm
   sudo modprobe nvidia_uvm
   ```

3. **清理GPU进程**
   ```bash
   # 强制终止占用GPU的进程
   sudo pkill -f isaac
   sudo pkill -f python3
   ```

### 2. Isaac Lab API兼容性问题

#### **问题描述**
```
AttributeError: type object 'FrameTransformerCfg' has no attribute 'FrameVisualizerCfg'
```

#### **解决方案**
已在最新代码中修复，使用简化配置：
```python
debug_vis=False  # 而不是 visualizer_cfg=...
```

### 3. 环境依赖问题

#### **检查Isaac Lab环境**
```bash
# 确认在正确的环境中
echo $ISAAC_LAB_PATH
echo $ISAACSIM_PATH

# 重新激活环境
cd /home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab
source isaaclab.sh
```

#### **检查Python依赖**
```bash
# 检查关键库
python -c "import warp; print('Warp OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import omni; print('Omniverse OK')"
```

## 🔧 **调试技巧**

### 1. 逐步测试
```bash
# 1. 先测试最简单的Isaac Lab示例
cd /home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab
./isaaclab.sh -p source/standalone/environments/cart_pole.py --headless

# 2. 再测试我们的状态机
cd /home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_clone
./run_state_machine.sh --headless
```

### 2. 日志分析
```bash
# 查看详细错误日志
./run_state_machine.sh --headless 2>&1 | tee debug.log

# 查看Isaac Sim日志
ls -la /home/larry/NVIDIA_DEV/isaac-sim/isaac-sim-standa/kit/logs/Kit/Isaac-Sim/
```

### 3. 环境变量检查
```bash
# 确认CUDA相关环境变量
echo $CUDA_VISIBLE_DEVICES
echo $LD_LIBRARY_PATH | grep cuda
```

## 📋 **重启后的完整检查清单**

### 步骤1: 系统状态检查
- [ ] `nvidia-smi` 显示GPU正常
- [ ] `ps aux | grep isaac` 无残留进程
- [ ] 系统温度正常，无过热

### 步骤2: 环境验证
- [ ] Isaac Lab环境激活
- [ ] Python依赖库正常导入
- [ ] USD文件路径正确

### 步骤3: 功能测试
- [ ] Isaac Lab示例运行正常
- [ ] 状态机headless模式测试
- [ ] 状态机可视化模式测试
- [ ] MediaPipe模式测试 (可选)

### 步骤4: 性能优化
- [ ] 调整position_threshold参数
- [ ] 优化状态等待时间
- [ ] 测试多环境并行

## 🆘 **紧急情况处理**

### 如果系统完全无响应
```bash
# 1. 强制重启
sudo reboot now

# 2. 检查硬件状态
sudo dmesg | grep -i error
sudo journalctl -xe | grep -i nvidia

# 3. 重新安装NVIDIA驱动 (最后手段)
sudo apt purge nvidia-*
sudo apt autoremove
sudo apt update
sudo apt install nvidia-driver-570
sudo reboot
```

### 如果Isaac Lab损坏
```bash
# 重新克隆Isaac Lab
cd /home/larry/NVIDIA_DEV/isaac_grasp_ws/
mv IsaacLab IsaacLab_backup
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

## 📞 **联系信息**

如果遇到无法解决的问题，请：
1. 检查Isaac Lab官方文档
2. 查看GitHub Issues
3. 记录详细错误日志进行分析 