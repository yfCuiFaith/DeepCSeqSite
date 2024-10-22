# DeepCSeqSite
DeepCSeqSite (DCS-SI) is a toolkit for protein-ligand binding sites prediction.<br>
The current version is a demo for DCS-SI.<br>
The formal version will be released later.<br>

# Quick Start
## Requirments
Platform = Linux<br>
Python = 2.7.x<br>
1.5.0 <= TensorFlow <= 1.10.0<br>
We strongly recommend you execute DCS-SI with GPU.
## Usage
Enter the root dir. If needed, you can get help information by this command:
```bash
python dcs_si.py -h
```
The demo contains three versions of model which differ in their network.<br>
The versions include DCS-SI-std, DCS-SI-k9, DCS-SI-k9a and DCS-SI-en.<br>
For example, you can load DCS-SI-std by:
```bash
python dcs_si.py --model DCS-SI-std
```
DCS-SI-std is the default version of DCS-SI. The kernel width in DCS-SI is k = 5.<br>
k9 means the kernel width k = 9, and 'a' in k9a means the model is trained on the augmented training set.<br>
DCS-SI-en is the enhanced version of DCS-SI, which executes forward propagation twice and takes the previous output into consideration.<br>

We provides all the test sets used in our paper.<br>
The test sets include SITA, SITA-EX1, SITA-EX2 and SITA-EX3.<br>
After loading a version of the model, you can easily test the model on the test sets with the guide of program. 
# Training a New Model
The source code and notes of the network architecture can be found in Models directory.<br>
Complete source code of training will be released later.
# License
DeepCSeqSite is [GPL 3.0-licensed](https://github.com/yfCuiFaith/DeepCSeqSite/blob/master/LICENSE)
