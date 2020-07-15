# MobileNetV3
pytorch implementation of MobileNetV3

This is a pytorch implementation of MobileNetV3,which includes MobileNetV3_large and MobileNetV3_small.I pre-trained this model with oxFolower datasets of 17 classes.You can 
execute inference using my pre-trained weights or train your own datasets.

## Inference:
  inference.py provides a class named 'Detector' for inference.You can initialize a Detector object and call it's 'detector' function to execute inference.The parameters of this
function are weight_path and picture_path,an example of inference are as below:

  ```python
  detector=Detector('large',num_classes=17)
  detector.detect('./weights/best.pkl','./1.jpg')
  ```
## Train model on your own datasets:
  Pictures for training should be put in 'data' folder.Split your data to several folders,the name of these folders should be named from '0' to num_classes(just follow this project)
then put them in 'data/splitData/train'.Note that the 'test' and 'valid' folder are not used in this project.If you need to execute testing or validation,you can modify this module.
  After preparing your dataset,You can choose which model to train in train.py,line 55:
  
   ```python
  net=MobileNetV3_large(num_classes=17)
  net=MobileNetV3_small(num_classes=17)
  ```
  You can also alternate the epoches and learning rate in the head of this file.
  After choosing the model you want to train and set the classes of your dataset,then run train.py to train.The weights will be saved as weights/last.pkl and weights/best.pkl.
  
  
  ## This project is a rough implentation of MobileNetV3,you can use it as the backbone of other networks or modify it for your propose.
