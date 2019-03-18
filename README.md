# CycleGAN and pix2pix in PyTorch

Our group has applicated our dataset (satellite) on the Pix2pix and CycleGAN which were written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung), and supported by [Tongzhou Wang](https://ssnl.github.io/). 

These are the informations of the authors:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)

Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/MLQZJ/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.

### Preparation of Dataset for Pix2pix and CycleGAN
- Download the dataset from the internet to your computer:
http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html
  - In our work, we use the datasets of Vaihingen. We download the 'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE' under the .datasets/images and 'ISPRS_semantic_labeling_Vaihingen' under the ./datasets/labels. We pre-treat the dataset by the command below to obtain the paired dataset: 

```bash
  python Image_Preprocessing.py 
```
  
- To view training results and loss plots, run
  ```bash
  python3 -m visdom.server
  ```
  and click the URL http://localhost:8097. 
  Note: Use another terminal and type "ssh -N -L localhost:8096:localhost:8097 vm_adress" when we use the virtual machine and click on     http://localhost:8096.
 
### CycleGAN train/test

- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/satellite --name satellite_cyclegan --model cycle_gan --dataset_mode aligned
```
  - Note: The code default is using the unaligned code for cyclegan if we don't add "--dataset_mode aligned". We use the aligned images 
    train the model, but we can also use the unaligned ones.
To see more intermediate results, check out `./checkpoints/satellite_cyclegan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/satellite --name satellite_cyclegan --model cycle_gan --dataset_mode aligned
```
- The test results will be saved to a html file here: `./results/satellite_cyclegan/latest_test/index.html`.

### pix2pix train/test
- Download and pre-treat the dataset as above.
  - Note: For pix2pix, it is necessary to have the paired image
- Train a model:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/satellite --name satellite_pix2pix --model pix2pix --direction BtoA
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out  `./checkpoints/satellite_pix2pix/web/index.html`.

- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/satellite --name satellite_pix2pix --model pix2pix --direction BtoA
```
- The test results will be saved to a html file here: `./results/satellite_pix2pix/test_latest/index.html`. You can find more scripts at `scripts` directory.

## [Datasets](docs/datasets.md)
Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Citation

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```
