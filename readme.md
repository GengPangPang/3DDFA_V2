# 3DDFA_V2三维人脸重建与OpenGL多视角渲染

本项目基于`3DDFA_V2`实现单张图像三维人脸重建，并使用`PyOpenGL/pyrender`对导出的`OBJ`三维人脸模型进行离屏渲染和多角度可视化。

## 主要功能

- 使用`3DDFA_V2`从单张人脸图像重建三维人脸模型
- 导出`OBJ`格式的三维人脸网格
- 生成三维投影图、深度图、姿态图和稠密关键点图
- 使用`OpenGL`对`OBJ`模型进行多角度离屏渲染
- 支持统一材质渲染和OBJ顶点颜色渲染
- 输出正面、左侧、右侧、俯视和斜视等视角结果

---

## 1.环境配置

### 1.1创建Conda环境

```bash
conda create -n face_landmark python=3.10 -y
conda activate face_landmark
```

### 1.2安装PyTorch和OpenCV

本项目使用的是`PyTorch2.4.0+CUDA12.1`。

```bash
pip install opencv-python-headless

pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

检查PyTorch和CUDA版本：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

如果输出类似下面结果，说明安装成功：

```text
2.4.0+cu121
12.1
```

---

## 2.下载项目代码

```bash
git clone https://github.com/cleardusk/3DDFA_V2.git
cd 3DDFA_V2
```

---

## 3.安装依赖

由于当前环境中已经手动安装了`torch`、`torchvision`和`opencv-python-headless`，因此不建议直接完整执行：

```bash
pip install -r requirements.txt
```

否则可能会覆盖已有的CUDA版PyTorch环境。

建议只补充缺失依赖：

```bash
pip install imageio imageio-ffmpeg cython scikit-image onnxruntime gradio
pip install onnx
```

---

## 4.编译3DDFA_V2扩展模块

```bash
sh ./build.sh
```

如果编译时报错：

```text
nms/cpu_nms.pyx:25:23:Invalid type.
cdef np.ndarray[np.int_t,ndim=1] order=scores.argsort()[::-1]
```

可以将`np.int_t`替换为`np.intp_t`：

```bash
sed -i 's/np.int_t/np.intp_t/g' FaceBoxes/utils/nms/cpu_nms.pyx
```

如果运行时报错：

```text
AttributeError:module 'numpy' has no attribute 'int'
```

可以将`dtype=np.int`替换为`dtype=int`：

```bash
sed -i 's/dtype=np.int)/dtype=int)/g' FaceBoxes/utils/nms/cpu_nms.pyx
```

然后清理旧的编译文件并重新编译：

```bash
rm -f FaceBoxes/utils/nms/cpu_nms.c
rm -f FaceBoxes/utils/nms/cpu_nms*.so
rm -rf FaceBoxes/utils/build
sh ./build.sh
```

---

## 5.运行3DDFA_V2三维人脸重建

### 5.1生成三维投影结果

```bash
python demo.py -f examples/inputs/emma.jpg -o 3d --onnx
```

输出文件：

```text
examples/results/emma_3d.jpg
```

### 5.2导出OBJ三维人脸模型

```bash
python demo.py -f examples/inputs/emma.jpg -o obj --onnx
```

输出文件：

```text
examples/results/emma_obj.obj
```

### 5.3生成深度图

```bash
python demo.py -f examples/inputs/emma.jpg -o depth --onnx
```

输出文件：

```text
examples/results/emma_depth.jpg
```

### 5.4生成姿态图

```bash
python demo.py -f examples/inputs/emma.jpg -o pose --onnx
```

输出文件：

```text
examples/results/emma_pose.jpg
```

### 5.5生成稠密关键点图

```bash
python demo.py -f examples/inputs/emma.jpg -o 2d_dense --onnx
```

输出文件：

```text
examples/results/emma_2d_dense.jpg
```

---

## 6.配置OpenGL离屏渲染环境

本项目在`WSL`环境下运行，没有图形界面，因此使用`OSMesa`进行OpenGL离屏渲染。

### 6.1安装Python渲染依赖

```bash
pip install trimesh pyrender pillow pyglet
```

### 6.2安装OSMesa

```bash
sudo dpkg --configure -a
sudo apt update
sudo apt install -y libosmesa6 libosmesa6-dev
```

### 6.3安装兼容版本的PyOpenGL

```bash
pip uninstall -y PyOpenGL PyOpenGL-accelerate
pip install PyOpenGL==3.1.4
```

---

## 7.运行OpenGL多视角渲染

项目中使用`render_opengl_obj_color.py`脚本读取`3DDFA_V2`生成的`OBJ`模型，并进行多角度离屏渲染。

### 7.1统一材质渲染

如果希望使用统一材质颜色渲染模型，可以运行：

```bash
python render_opengl_obj_color.py \
  --obj examples/results/emma_obj.obj \
  --out_dir outputs/opengl_renders \
  --bg_mode light \
  --color_mode material
```

输出目录：

```text
outputs/opengl_renders/
```

输出文件包括：

```text
front.png
left.png
right.png
top.png
iso.png
multi_view_grid.png
```

### 7.2顶点颜色渲染

3DDFA_V2导出的OBJ文件中，顶点信息可能包含从原图采样得到的颜色，例如：

```text
v 1725.41 1464.67 140.24 0.57 0.40 0.32
```

其中前三列是三维坐标，后三列是RGB颜色值。  
如果希望使用OBJ中的顶点颜色进行渲染，可以运行：

```bash
python render_opengl_obj_color.py \
  --obj examples/results/emma_obj.obj \
  --out_dir outputs/opengl_renders_color \
  --bg_mode light \
  --color_mode vertex
```

输出目录：

```text
outputs/opengl_renders_color/
```

输出文件包括：

```text
front.png
left.png
right.png
top.png
iso.png
multi_view_grid.png
```

### 7.3自动颜色模式

也可以使用`auto`模式。该模式会自动检测OBJ文件中是否包含有效顶点颜色：

- 如果OBJ包含有效顶点颜色，则使用顶点颜色渲染
- 如果OBJ不包含有效顶点颜色，则自动回退为统一材质渲染

运行命令：

```bash
python render_opengl_obj_color.py \
  --obj examples/results/emma_obj.obj \
  --out_dir outputs/opengl_renders_auto \
  --bg_mode light \
  --color_mode auto
```

### 7.4渲染参数说明

`render_opengl_obj_color.py`支持以下主要参数：

```text
--obj          3DDFA_V2导出的OBJ模型路径
--out_dir      渲染结果输出目录
--size         输出图像尺寸，默认512
--bg_mode      背景颜色，可选light、dark、white
--color_mode   渲染颜色模式，可选auto、vertex、material
```

其中：

- `--color_mode auto`：自动检测并优先使用OBJ顶点颜色
- `--color_mode vertex`：强制使用OBJ顶点颜色
- `--color_mode material`：使用统一材质颜色
- `--bg_mode light`：浅灰色背景
- `--bg_mode dark`：深色背景
- `--bg_mode white`：白色背景

---

## 8.完整运行流程示例

进入项目目录并激活环境：

```bash
conda activate face_landmark
cd ~/projects/3DDFA_V2
```

运行3DDFA_V2生成结果：

```bash
python demo.py -f examples/inputs/emma.jpg -o 3d --onnx
python demo.py -f examples/inputs/emma.jpg -o obj --onnx
python demo.py -f examples/inputs/emma.jpg -o depth --onnx
python demo.py -f examples/inputs/emma.jpg -o pose --onnx
python demo.py -f examples/inputs/emma.jpg -o 2d_dense --onnx
```

运行OpenGL统一材质渲染：

```bash
python render_opengl_obj_color.py \
  --obj examples/results/emma_obj.obj \
  --out_dir outputs/opengl_renders \
  --bg_mode light \
  --color_mode material
```

运行OpenGL顶点颜色渲染：

```bash
python render_opengl_obj_color.py \
  --obj examples/results/emma_obj.obj \
  --out_dir outputs/opengl_renders_color \
  --bg_mode light \
  --color_mode vertex
```

查看输出结果：

```bash
find outputs/opengl_renders -type f
find outputs/opengl_renders_color -type f
```

如果在WSL中希望用Windows文件管理器打开输出目录，可以运行：

```bash
explorer.exe outputs/opengl_renders
explorer.exe outputs/opengl_renders_color
```

---

## 9.最终输出文件

```text
examples/results/
├── emma_3d.jpg
├── emma_obj.obj
├── emma_depth.jpg
├── emma_pose.jpg
└── emma_2d_dense.jpg

outputs/opengl_renders/
├── front.png
├── left.png
├── right.png
├── top.png
├── iso.png
└── multi_view_grid.png

outputs/opengl_renders_color/
├── front.png
├── left.png
├── right.png
├── top.png
├── iso.png
└── multi_view_grid.png
```

---

## 10.说明

本项目主要完成任务8中的三维人脸重建与渲染可视化部分。项目没有重新训练`3DDFA_V2`模型，而是使用其现成模型进行推理，并重点完成从单张图像到三维人脸模型导出，再到OpenGL多视角渲染的完整流程。

其中，OpenGL渲染部分支持两种方式：

- 统一材质渲染：更适合观察三维人脸网格的几何结构
- 顶点颜色渲染：能够保留OBJ模型中的原图采样颜色，视觉效果更接近真实人脸外观