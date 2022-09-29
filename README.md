# INDIG

##### Contributors: [Kwang Hee Lee](https://lekwanghee.github.io/) .

> Kwang Hee Lee and Myoung Ho Kim, "Bayesian inductive learning in group recommendations for seen and unseen groups", Information Sciences 610 (2022): 725-745

This repository contains a PyTorch implementation of INDIG 


### Requirements
The code has been tested running under Python 3.8 with Jupyter Notebook on Windows 10:


### Input Format



### Repository Organization
- ``data/`` contains the necessary input file(s) for each dataset in the specified format.
- ``model/`` contains:
    - INDIG model (``models.py``);    


### Running INDIG

Note: The model is not deterministic. All the experimental results provided in the paper are averaged across multiple
 runs.
 
 
## Reference
 If you make use of this code or the INDIG algorithm in your work, please cite the following paper:

```
@article{lee2022bayesianindig,
  title={Bayesian inductive learning in group recommendations for seen and unseen groups},
  author={Lee, Kwang Hee and Kim, Myoung Ho},
  journal={Information Sciences},
  volume={610},
  pages={725--745},
  year={2022},
  issn = {0020-0255},
  publisher={Elsevier}
}
```
