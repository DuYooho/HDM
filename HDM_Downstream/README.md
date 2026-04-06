# Downstream Polyp Segmentation with HDM

## Dataset Preparation

### 1. 2D Dataset
- Download the 2D dataset from [this repository](https://github.com/DengPingFan/PraNet).
- Download the extracted features by HDM from [this link](https://pan.baidu.com/s/145u8ffMMM_03lKfRgIB8KQ?pwd=347d).
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
- Download the extracted features by HDM from [this link](https://pan.baidu.com/s/14Dlr9zcwMEyernAASMW71Q?pwd=rs78).
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
| PraNet+Ours      | [Download](https://pan.baidu.com/s/1-tVn2VKLzvmV96RQ0sJa-g?pwd=enr8) |
| FCBFormer+Ours   | [Download](https://pan.baidu.com/s/1ASofKTqB2xOE0-fYA8_meQ?pwd=jyg9) |
| Polyp-PVT+Ours   | [Download](https://pan.baidu.com/s/1JDUARwx7midG6sKA0BQsQg?pwd=6icx) |

**(2) SUN-SEG Dataset Models**

| Methods          | Link                                                                                                                                                  |                                      
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| FCBFormer+Ours   | [Download](https://pan.baidu.com/s/1Y3jLXyagzC1lGV6XWY6rrQ?pwd=qef5)      |
| Polyp-PVT+Ours   | [Download](https://pan.baidu.com/s/1bjiYmEBRrDR4GunirsHUBQ?pwd=cssx)      |

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