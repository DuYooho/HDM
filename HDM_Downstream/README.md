# Downstream Polyp Segmentation with HDM

## Dataset Preparation

### 1. 2D Dataset
- Download the 2D dataset from [this repository](https://github.com/DengPingFan/PraNet).
- Download the extracted features by HDM from [this link](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EV2ilPTFJpNFnV4riXwD9swBiKpF7ig_m-h5n02L__LlHA?e=DPvtZW).
- Organize the dataset with the following structure:

  ```bash
  ├── ${data_root}
  │   ├── ${train_data_dir}
  │   │   ├── images
  │   │   │   ├── ***.png
  │   │   ├── masks
  │   │   │   ├── ***.png
  │   │   ├── features
  │   │   │   ├── ***.pt
  │   ├── ${test_data_dir}
  │   │   ├── images
  │   │   │   ├── ***.png
  │   │   ├── masks
  │   │   │   ├── ***.png
  │   │   ├── features
  │   │   │   ├── ***.pt
  ```

### 2. SUN-SEG Dataset
- Download the SUN-SEG dataset from [this repository](https://github.com/GewelsJI/VPS).
- Download the extracted features by HDM from [this link](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/Ee2X8wZlHPxDhPkJ2bJdMF8Ba07Rx-BX0naZAwVc8Wqqsg?e=6y6MGy).
- Organize the dataset with the following structure:

  ```bash
  ├── ${data_root}
  │   ├── ${train_data_dir}
  │   │   ├── Frame
  │   │   │   ├── case***
  │   │   │   │   ├── ***.jpg
  │   │   ├── GT
  │   │   │   ├── case***
  │   │   │   │   ├── ***.png
  │   │   ├── Features
  │   │   │   ├── case***
  │   │   │   │   ├── ***.pt
  │   ├── ${test_data_dir}
  │   │   ├── Frame
  │   │   │   ├── case***
  │   │   │   │   ├── ***.jpg
  │   │   ├── GT
  │   │   │   ├── case***
  │   │   │   │   ├── ***.png
  │   │   ├── Features
  │   │   │   ├── case***
  │   │   │   │   ├── ***.pt
  ```

- For the SUN-SEG dataset, use only the **Unseen** test data from both the **Easy** and **Hard** cases. Combine them into a new test set in the directory specified by `${test_data_dir}`.

---

## Train, Test, and Evaluate Models with HDM Features

### 1. Configuration
Specify the `workspace`, `MODEL_NAME`, and `_DATA_TYPE` in the `global_config.py` file. (Note: In our setup, all datasets, models, logs, and results are stored in the folder specified by `workspace`, but you can adjust this setting as needed.)

### 2. Train Your Own Models
To train your models using HDM features, run the following command:

```bash
python train.py
```

### 3. Test Trained Models

- To test models trained on the **2D dataset**, run:

  ```bash
  python test.py
  ```

- To test models trained on the **SUN-SEG dataset**, run:

  ```bash
  python test_SUN-SEG.py
  ```

---

#### Model Zoo

We provide pre-trained models for both the **2D Dataset** and **SUN-SEG** dataset.

**(1) 2D Dataset Models**

| Methods          | Link                                                                                                                                             |                                      
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| PraNet+Ours      | [Download](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EQnz3XB0s1JFijYPRJa1sCgB35Ma7cm4lCot8nliJ28PGQ?e=mHDtDx) |s
| FCBFormer+Ours   | [Download](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EZ6YsvCgOrhNiwFHcG3_2oMBP-Ez3sOtVMUiBtfmpRkOJQ?e=WKzKXf) |
| Polyp-PVT+Ours   | [Download](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/Ef8og3Fm0d5EkmBakiXQzYYBA54lyHodnbH3h3vypA_qxA?e=dBPpDS) |

**(2) SUN-SEG Dataset Models**

| Methods          | Link                                                                                                                                                  |                                      
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCBFormer+Ours   | [Download](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EYS0S8slZ1xGhKOTyqxlojMBwk0y87DHHgYNvRZ0pMUd-Q?e=KpykhJ)      |
| Polyp-PVT+Ours   | [Download](https://cuhko365-my.sharepoint.com/:u:/g/personal/118010053_link_cuhk_edu_cn/EampDpyWyzlHu3tyzw65-_oBZoS7lQpUM9p-esdbw97Iyg?e=2D73Zz)      |

You can download the pre-trained weights or train the models yourself.

---

### 4. Evaluation

To evaluate models:

- For models trained on the **2D dataset**, run:

  ```bash
  python eval.py
  ```

- For models trained on the **SUN-SEG dataset**, run:

  ```bash
  python eval_SUN-SEG.py
  ```