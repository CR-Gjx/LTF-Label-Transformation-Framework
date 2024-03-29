# LTF-Label-Transformation-Framework
The codes of paper "LTF: A Label Transformation Framework for Correcting Target Shift" on ICML2020.

## Requirements

* PyTorch
* Python 3.6.5


## Training
To train the LTF models, you need to download the pre-trained generative model in the https://drive.google.com/file/d/1PaeHIvjF8VFz2_kfiUrbl49pOiuyH4bh/view?usp=sharing, and unzip it as **result** folder.

To run the experiment for the fashion-mnist dataset:
```
$ python main.py --dataset='f-m'  --num_class=10 --c_epochs=20
```
dataset:

* f-m: fashion mnist 
* mnist: mnist
* cifar10
 
num_class: The number of classes is 10


c_epochs: The training epochs for the classifier.

tweak:  target shift setting
* 0: **Random Dirichlet Shift** In this shift, we randomly gener-ate a label distributionPTYby employing the Dirichletdistribution with different values of the concentration parameter α.
* 1: **Tweak-One Shift** To evaluate the performance on the largelabel probability quantification.  In our experiments,the ratio of one class is set to[0.5,0.6,0.7,0.8,0.9],respectively, while ratios of other classes are uniform,
* 2: **Minority-Class Shift** To evaluate the performance on thesmall label probability quantification.  In our experi-ments,[20%,30%,40%,50%]classes are set to 0.001,respectively, while ratios of other classes are uniform.

The results are shown as the output file, e.g. outputf-m010.txt


To run the experiment for the synthetic regression dataset:
```
$ python label_shift_regression.py --id=1
```

id:  target shift setting
* 1: **Left Gaussian** The Gaussian with the mean -0.707.
* 2: **Right Gaussian** The Gaussian with the mean 0.707.
* 3: **Mix Gaussian** The Mix Gaussian of 1 and 2.
* 4: **Random** The target distribution is generated by a random network.

## Reference
```bash
@inproceedings{guo2020ltf,
  title={LTF: A Label Transformation Framework for Correcting Label Shift},
  author={Guo, Jiaxian and Gong, Mingming and Liu, Tongliang and Zhang, Kun and Tao, Dacheng},
  booktitle={International Conference on Machine Learning},
  pages={3843--3853},
  year={2020},
  organization={PMLR}
}
```
