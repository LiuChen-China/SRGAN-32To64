### 描述
利用python的tensorflow.keras模块搭建超分对抗网(SRGAN)，将32x32的图片转换为64x64的图片  
训练模型的脚本是Module/Train.py(没有数据，无法直接运行)，生成模型保存在Static/generator.h5
### 研究笔记
+ 生成器整体是一个残差网络，而内部又存在多个子残差块
+ 残差块的最终输出应为原始输入的副本+网络输出
+ 判别器的学习率应大于生成器
### 环境
python3  
tensorflow-gpu==1.13.0 or tensorflow==1.13.0  
### 效果
从左到右是nearest算法、SRGAN、原图生成的图片  、
![Alt text]()
