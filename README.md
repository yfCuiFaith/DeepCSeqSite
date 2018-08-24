# DeepCSeqSite
DeepCSeqSite (DCS-SI) is a toolkit for protein-ligand binding sites prediction.<br>
The current version is a demo for DCS-SI. The formal version will be released later.
# Quick Start
## Requirments
TensorFlow >= 1.5.0
## Usage
Enter the dir of DCS-SI. If needed, you can get help information by this command:
```bash
python dcs_si.py -h
```
The demo contains three versions of model which differ in their network. The versions include DCS-SI-std, DCS-SI-k9 and DCS-SI-k9a. For example, you can run DCS-SI-std by:
```bash
python dcs_si.py --model DCS-SI-std
```
