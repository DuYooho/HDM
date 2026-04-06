# Highlighted Diffusion Model (HDM)

This repository contains the official implementation of the paper **[Highlighted Diffusion Model as Plug-in Priors for Polyp Segmentation](https://ieeexplore.ieee.org/document/10733999)**, presented at JBHI-2024.

## Table of Contents
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Sampling with HDMs](#sampling-with-hdms)
- [Training Your Own HDMs](#training-your-own-hdms)
- [Downstream Evaluation](#downstream-evaluation)
- [Citations](#citations)

## Requirements

To get started, set up your environment with the following dependencies:

```bash
conda create -n HDM python=3.10
conda activate HDM
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch-lightning==1.9.3 omegaconf einops tensorboard albumentations natsort prettytable timm==0.9.5 thop
```

## Dataset Preparation

### 1. 2D Dataset
- Download the 2D dataset from [this repository](https://github.com/DengPingFan/PraNet).
- Download highlighted images from [this link](https://pan.baidu.com/s/1yGSmhdc08quiw29s5Vel7A?pwd=hag2).
- Organize the dataset as follows:

  ```bash
  ├── ${data_root}
  │   ├── ${train_data_dir}
  │   │   ├── images
  │   │   │   ├── ***.png
  │   │   ├── masks
  │   │   │   ├── ***.png
  │   │   ├── highlighted_GT
  │   │   │   ├── ***.png
  │   ├── ${test_data_dir}
  │   │   ├── images
  │   │   │   ├── ***.png
  │   │   ├── masks
  │   │   │   ├── ***.png
  │   │   ├── highlighted_GT
  │   │   │   ├── ***.png
  ```

### 2. SUN-SEG Dataset
- Download the dataset from [this repository](https://github.com/GewelsJI/VPS).
- Download highlighted images from [this link](https://pan.baidu.com/s/1iYKtaw86IGPhgaGTZC4OlA?pwd=u4g3).
- Organize the dataset as follows:

  ```bash
  ├── ${data_root}
  │   ├── ${train_data_dir}
  │   │   ├── Frame
  │   │   │   ├── case***
  │   │   │   │   ├── ***.jpg
  │   │   ├── GT
  │   │   │   ├── case***
  │   │   │   │   ├── ***.png
  │   │   ├── Highlighted_GT
  │   │   │   ├── case***
  │   │   │   │   ├── ***.png
  │   ├── ${test_data_dir}
  │   │   ├── Frame
  │   │   │   ├── case***
  │   │   │   │   ├── ***.jpg
  │   │   ├── GT
  │   │   │   ├── case***
  │   │   │   │   ├── ***.png
  │   │   ├── Highlighted_GT
  │   │   │   ├── case***
  │   │   │   │   ├── ***.png
  ```

- For the **SUN-SEG** dataset, combine the **Unseen** test data from both the **Easy** and **Hard** cases into a new test set under `test_data_dir`.

## Sampling with HDMs

### Model Zoo

Pre-trained HDM models are available for download:

| Methods       | Download Link                                                                                                                                    |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| HDM_2D_Polyp  | [Download](https://pan.baidu.com/s/1ePsYcIhhANukxr82G7sO1Q?pwd=agav) |
| HDM_SUN-SEG   | [Download](https://pan.baidu.com/s/1k-7vpbeP5bfIUi8lBQiE_w?pwd=2bca) |

### Running Inference

To run inference using HDMs:

1. Set `CONFIG_FILE_PATH` and `RESUME_PATH` in the `sample.py` file.
2. Run the following command:

   ```bash
   python sample.py
   ```

### Extracted Features

Extracted features are available for download:

| Methods       | Download Link                                                                                                                                         |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| HDM_2D_Polyp  | [Download](https://pan.baidu.com/s/145u8ffMMM_03lKfRgIB8KQ?pwd=347d)      |
| HDM_SUN-SEG   | [Download](https://pan.baidu.com/s/14Dlr9zcwMEyernAASMW71Q?pwd=rs78)      |

## Training Your Own HDMs

To train your own HDM models:

1. Set `train_data_dir` and `test_data_dir` in the corresponding configuration file (`HDM_xxx.yaml`) located in the `configs` folder.
2. Set `CONFIG_FILE_PATH` in the `main.py` file.
3. Run the following command:

   ```bash
   python main.py
   ```

## Downstream Evaluation

For downstream evaluation, detailed instructions can be found in the `HDM_Downstream` folder. Please refer to [this file](HDM_Downstream/README.md) for further details.

## Citations

If you find **HDM** useful for your research, please cite our paper:

```bibtex
@ARTICLE{du2024highlighted,
  author={Du, Yuhao and Jiang, Yuncheng and Tan, Shuangyi and Liu, Si-Qi and Li, Zhen and Li, Guanbin and Wan, Xiang},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Highlighted Diffusion Model as Plug-in Priors for Polyp Segmentation}, 
  year={2024},
  doi={10.1109/JBHI.2024.3485767}}
```
