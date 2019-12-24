# dazhuanocr
大篆字体识别的一个demo,利用卷积神经网络对16个大篆字体进行识别，包括一个小的数据集，以及投影法字符分割代码，项目可对一个包含大篆字体的图片上的字符进行分割，并分别送入神经网络中进行识别
此项目解释器为python3.7
所用的库有
sklearn==0.0
tensorflow==2.0.0rc0
numpy==1.17.0
opencv==3.4.5
各文件介绍：
│  datamaking.py  数据整理，图片处理，去噪，将图片变为白底黑字
│  dazhuan_app.py  最后用来测试效果的，可以修改里面的图片文件
│  inference.py   推理代码  
│  train.py       训练代码
│  vertical_horizontal_simplecut.py  投影法字符分割代码
│  
├─.idea
│  │  misc.xml
│  │  modules.xml
│  │  workspace.xml
│  │  大篆字体识别.iml
│  │  
│  └─inspectionProfiles
│          profiles_settings.xml
│          
├─checkpoint   保存的模型文件夹，可进行断点续训
│      checkpoint
│      dazhuan.tf.data-00000-of-00001
│      dazhuan.tf.index
│      
├─cutimg  分割后的图片
├─x_test  测试图片
├─杂图
运行dazhuan_app.py显示图片分割和识别结果
