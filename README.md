# 此demo有两个部分
1.cuda
测试cuda是否安装以及版本信息
2.libtorch
测试能否加载模型，此处使用yolov8转换pt模型为torchscripts模型

## cuda测试
见对应文件夹内描述即可。

## libtorch测试
### 首先要安装libtorch
这里直接官网下载libtorch的zip文件，直接解压到/usr/local/下，得到libtorch。即为/usr/local/libtorch

### 测试库
除了libtorch，还需要gflag，glog，opencv
其中libtorch位置需要单独设定，见CMakeLists.txt

### 测试文件

1.test.cc
测试torch能否调用cuda

2.run_model.cc
调用ObjectDetector类，输入4个参数：图片文件夹路径,模型路径,绘制矩形框后的图片保存路径,是否绘制矩形框
自己训练的模型需要修改传入ObjectDetector类的classes

### 使用
cd libtorch
mkdir build && cd build
cmake ..
make