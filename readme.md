# Efficient Privacy Auditing in Federated Learning (USENIX Security 2024)

This repository contains the official code for the paper **"Efficient Privacy Auditing in Federated Learning"**, published at the **33rd USENIX Security Symposium (USENIX Security 2024)**. The code consists of two main parts:
1. **Run Federated Learning (FL) tasks**  
2. **Audit the privacy risks of the trained FL models**

We leverage the **FedML framework** (a PyTorch-based federated learning library) to facilitate the FL process. Our implementation includes modifications to the FedML codebase to save intermediate training information for auditing purpose.

---

## 📁 Code Structure

The repository includes the following key scripts:

| Script Name        | Description                                         |
|--------------------|-----------------------------------------------------|
| **1_create_split.py** | Prepares the federated learning data splits.        |
| **2_run_fl.py**      | Executes the federated learning training process.   |
| **3_run_audit.py**   | Audits the privacy risks of the trained model.      |

---

## 🖥️ Running the Code

To run the complete process (data preparation, federated learning, and privacy auditing), execute the provided bash script (which trains resnet56 model on CIFAR-10).

```bash
bash run.sh
```
For different models and dataset combinations, please refer to the Table 1 in the paper.

## 🚀 Requirements

Before running the code, ensure you have the following dependencies installed:
```
pip install -r requirements.txt
```

## 🔍 How to Cite

If you find this code useful in your research, please cite our paper:
```
@inproceedings {299655,
author = {Hongyan Chang and Brandon Edwards and Anindya S. Paul and Reza Shokri},
title = {Efficient Privacy Auditing in Federated Learning},
booktitle = {33rd USENIX Security Symposium (USENIX Security 24)},
year = {2024},
isbn = {978-1-939133-44-1},
address = {Philadelphia, PA},
pages = {307--323},
url = {https://www.usenix.org/conference/usenixsecurity24/presentation/chang},
publisher = {USENIX Association},
month = aug
}
```

## 📬 Contact
For questions or feedback, feel free to reach out to the authors via email or open an issue in this repository.

