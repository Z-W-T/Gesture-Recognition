* 基于ssd_mobilenetv3的手势识别（python）
如字面意义所示，整个模型主要由Mobilenetv3以及SSDlite两个部分构成。Mobilenetv3主要用于进行图像特征提取。SSD主要用于进行手部位置识别以及标签分类判断。

** Mobilenetv3
由于该网络目标设置于pad等移动端设备上进行使用，因此进行手势识别的网络需要具备在较差设备性能条件下快速运算的能力。Mobilenet就是针对这种场景设计的。

Mobilenet使用了深度可分离卷积提取特征图谱，即采用逐通道卷积与逐点卷积相结合的方式，能够有效减少参数数量，降低运算成本。

** SSDLite
SSD 全称 Single Shot MultiBox Detector，是一种用于目标检测的单次检测算法，每次可以检测多个物体。SSD 的移动端友好型变体，即 SSDlite能够快速高效的实现SSD的功能，从而更加适用于嵌入设备中。

* 代码结构

** /GESTURE/main.py、model.py、CNN.py
这几个部分是一个简单的Alexnet。
CNN.py中详细记录了网络的层次结构（归一化-->卷积-->池化-->卷积-->池化-->flatten(向量展平)-->dropout(随机丢弃)-->全连接-->dropout-->全连接）
model.py主要将样本进行分批处理，然后调用CNN中的模型进行训练与测试
main.py作为主函数，读入数据。初始化后将数据传入model中

** /GESTURE/HGDmaster
这个部分采用ssd_mobilenetv3构建完整模型并且针对Dataset中的[ann_test,test]、[ann_subsample,subsample]两个数据集进行训练以及测试。

/GESTURE/HGDmaster/classifier 该文件夹中所有文件均为库文件供/detector当中的文件使用
/GESTURE/HGDmaster/detector/model.py、ssd_mobilenetv3.py 这两个文件用于实现ssd_mobilenetv3神经网络细节
/GESTURE/HGDmaster/demo.py 调用detector文件夹中的神经网络进行训练与测试。同时，通过/GESTURE/dataset/DataSet.py文件中的dataset类实现批量图片与标签的输入

** /GESTURE/dataset
文件夹中包含了3个数据集test，subsample，robusttest以及对应的标签文件。同时其中包含的DataSet.py文件用于批量读取图片与标签，便于神经网络进行训练。