# FineCIR: Explicit Parsing of Fine-Grained Modification Semantics for Composed Image Retrieval

This is an open-source implementation of the paper "FineCIR: Explicit Parsing of Fine-Grained Modification Semantics for Composed Image Retrieval" (**FineCIR**).

*PS: The full FineCIR dataset will be updated after acceptance.*


### Installation
1. Clone the repository

```sh
git clone https://github.com/SDU-L/FineCIR
```

2. Running Environment

```sh
Platform: NVIDIA A40 48G
Python  3.10.8
Pytorch  2.5.1
Transformers  4.25.0
```


### Data Preparation

#### 1. Fashion-domain Dataset


  - ##### Fine-FashionIQ
    First, download the FashionIQ dataset following the instructions in
    the [official repository](https://github.com/XiaoxiaoGuo/fashion-iq).

    After downloading the dataset, to obtain our proposed Fine-FashionIQ dataset, replace the ```captions``` folder with our provided ```FineMT```, ensure that the folder structure matches the following:
    ```
    ├── Fine-FashionIQ
    │   ├── captions
    |   |   ├── fiq.fine.dress.[train | val].json
    |   |   ├── fiq.fine.toptee.[train | val].json
    |   |   ├── fiq.fine.shirt.[train | val].json

    │   ├── image_splits
    |   |   ├── split.dress.[train | val].json
    |   |   ├── split.toptee.[train | val].json
    |   |   ├── split.shirt.[train | val].json

    │   ├── dress
    |   |   ├── [B000ALGQSY.jpg | B000AY2892.jpg | B000AYI3L4.jpg |...]

    │   ├── shirt
    |   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]

    │   ├── toptee
    |   |   ├── [B0000DZQD6.jpg | B000A33FTU.jpg | B000AS2OVA.jpg | ...]
    ```


  - ##### FashionIQ

    Download the FashionIQ dataset following the instructions in
    the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq).

    After downloading the dataset, ensure that the folder structure matches the following:

    ```
    ├── FashionIQ
    │   ├── captions
    |   |   ├── cap.dress.[train | val | test].json
    |   |   ├── cap.toptee.[train | val | test].json
    |   |   ├── cap.shirt.[train | val | test].json

    │   ├── image_splits
    |   |   ├── split.dress.[train | val | test].json
    |   |   ├── split.toptee.[train | val | test].json
    |   |   ├── split.shirt.[train | val | test].json

    │   ├── dress
    |   |   ├── [B000ALGQSY.jpg | B000AY2892.jpg | B000AYI3L4.jpg |...]

    │   ├── shirt
    |   |   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]

    │   ├── toptee
    |   |   ├── [B0000DZQD6.jpg | B000A33FTU.jpg | B000AS2OVA.jpg | ...]
    ```



#### 2. Open-domain Dataset

  - ##### Fine-CIRR

    First, download the CIRR dataset following the instructions in the [official repository](https://github.com/Cuberick-Orion/CIRR).

    After downloading the dataset, to obtain our proposed Fine-CIRR dataset, replace the ```captions``` folder with our provided ```FineMT```, ensure that the folder structure matches the following:

    ```

    ├── Fine-CIRR
    │   ├── train
    |   |   ├── [0 | 1 | 2 | ...]
    |   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

    │   ├── dev
    |   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

    │   ├── fine-cirr
    |   |   ├── finemt_captions
    |   |   |   ├── cirr.fine.[train | val].json
    |   |   ├── image_splits
    |   |   |   ├── split.rc2.[train | val].json
    ```


  - ##### CIRR

    Download the CIRR dataset following the instructions in the [**official repository**](https://github.com/Cuberick-Orion/CIRR).

    After downloading the dataset, ensure that the folder structure matches the following:

    ```
    ├── CIRR
    │   ├── train
    |   |   ├── [0 | 1 | 2 | ...]
    |   |   |   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]

    │   ├── dev
    |   |   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]

    │   ├── test1
    |   |   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]

    │   ├── cirr
    |   |   ├── captions
    |   |   |   ├── cap.rc2.[train | val | test1].json
    |   |   ├── image_splits
    |   |   |   ├── split.rc2.[train | val | test1].json
    ```



### Train Phase

1. Train FineCIR
```sh
python3 train.py
```

### Acknowledgement
Our implementation is based on [LAVIS](https://github.com/chiangsonw/cala?tab=readme-ov-file).

