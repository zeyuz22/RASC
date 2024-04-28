# Reconfigurability-Aware Selection for Contrastive Active Domain Adaptation

This repository is the official implementation of "Reconfigurability-Aware Selection for Contrastive Active Domain
Adaptation".

## Setup Datasets

- Download [Office-31 Dataset](https://faculty.cc.gatech.edu/~judy/domainadapt/)
- Download [Office-Home Dataset](http://hemanthdv.org/OfficeHome-Dataset/)
- Download [VisDA-2017 Dataset](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
- MiniDomainNet You need to download the DomainNet dataset first, and the MiniDomainNet's split files can be downloaded
  at this [google drive](https://drive.google.com/open?id=15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7).

  The data folder should be structured as follows, and we have provided text index files:
    ```
    ├── data/
    │   ├── office31/     
    |   |   ├── amazon/
    |   |   ├── dslr/
    |   |   ├── webcam/
    │   ├── office-home/
    |   |   ├── Art/
    |   |   ├── Clipart/
    |   |   ├── Product/
    |   |   ├── Real_World/
    │   ├── VisDA/
    |   |   ├── validation/
    |   |   ├── train/
    │   ├── domainnet/	
    |   |   ├── clipart/
    |   |   |—— infograph/
    |   |   ├── painting/
    |   |   |—— quickdraw/
    |   |   ├── real/	
    |   |   ├── sketch/	
    |   |——
    ```
  We have provided text index files.

## Training on Office-Home

- Pre-train model on the source domain by running:

```
python3 main.py --pre-train --lr 0.01 --batch-size 32 --epochs 30 --s <source_domain_name> --source <source_domain_dataset_path> --t <source_domain_dataset_path> --target <target_domain_dataset_path> --target-val <target_domain_dataset_path> --class-num 65
```

- Train RASC using the pre-trained model by running:

```
python3 main.py --lr 0.01 --batch-size 32 --epochs 30 --s <source_domain_name> --source <source_domain_dataset_path> --t <target_domain_name> --target <target_domain_dataset_path> --target-val <target_domain_dataset_path> --class-num 65
```

## Training on Office-31

- Pre-train model on the source domain by running:

```
python3 main.py --pre-train --lr 0.01 --batch-size 32 --epochs 50 --s <source_domain_name> --source <source_domain_dataset_path> --t <source_domain_dataset_path> --target <target_domain_dataset_path> --target-val <target_domain_dataset_path> --class-num 31
```

- Train RASC using the pre-trained model by running:

```
python3 main.py --lr 0.01 --batch-size 32 --epochs 30 --s <source_domain_name> --source <source_domain_dataset_path> --t <target_domain_name> --target <target_domain_dataset_path> --target-val <target_domain_dataset_path> --class-num 31
```

## Training on VisDA

- Pre-train model on the source domain by running:

```
python3 main.py --pre-train --lr 0.01 --batch-size 36 --epochs 1 --s syntatic --source <source_domain_dataset_path> --t real --target <target_domain_dataset_path> --target-val <target_domain_dataset_path> --class-num 12
```

- Train RASC using the pre-trained model by running:

```
python3 main.py --lr 0.01 --batch-size 36 --epochs 10 --s syntatic --source <source_domain_dataset_path> --t <target_domain_name> --target <target_domain_dataset_path> --target-val <target_domain_dataset_path> --class-num 12
```

## Training on MiniDomainNet

- Pre-train model on the source domain by running:

```
python3 main.py --pre-train --lr 0.01 --batch-size 32 --epochs 10 --s <source_domain_name> --source <source_domain_dataset_path> --t <source_domain_dataset_path> --target <target_domain_dataset_path> --target-val <target_domain_dataset_path> --class-num 126
```

- Train RASC using the pre-trained model by running:

```
python3 main.py --lr 0.01 --batch-size 32 --epochs 10 --s <source_domain_name> --source <source_domain_dataset_path> --t <target_domain_name> --target <target_domain_dataset_path> --target-val <target_domain_dataset_path> --class-num 126
```

## Acknowledgments

This project is based on the following open-source project. We thank the authors for making the source code publicly
available.

- [Transferable-Query-Selection](https://github.com/thuml/Transferable-Query-Selection)
