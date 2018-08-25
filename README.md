# DeepCSeqSite
DeepCSeqSite (DCS-SI) is a toolkit for protein-ligand binding sites prediction.<br>
The current version is a demo for DCS-SI.<br>
The formal version will be released later.<br>

# Quick Start
## Requirments
Python = 2.7.x
1.5.0 <= TensorFlow <= 1.10.0<br>
We strongly recommend you execute DCS-SI with GPU.
## Usage
Enter the root dir of DCS-SI. If needed, you can get help information by this command:
```bash
python dcs_si.py -h
```
The demo contains three versions of model which differ in their network.<br>
The versions include DCS-SI-std, DCS-SI-k9 and DCS-SI-k9a.<br>
For example, you can load DCS-SI-std by:
```bash
python dcs_si.py --model DCS-SI-std
```
DCS-SI-std is the default version of DCS-SI. The kernel width in DCS-SI is k = 5.<br>
k9 means the kernel width k = 9, and 'a' in k9a means the model is trained on the augmented training set.<br>

We provides all the test sets used in our paper.<br>
The test sets include SITA, SITA-EX1, SITA-EX2 and SITA-EX3.<br>
After loading a version of the model, you can easily test the model on the test sets with the guide of program. 
## License
DeepCSeqSite is [GPL 3.0-licensed](https://github.com/yfCuiFaith/DeepCSeqSite/blob/master/LICENSE)
