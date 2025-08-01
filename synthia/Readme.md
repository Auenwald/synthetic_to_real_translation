# 🧠 SYNTHIA Dataset Setup Guide

This guide explains how to download, extract, and organize the **SYNTHIA-RAND-CITYSCAPES** and **synthiastyle** datasets for semantic segmentation tasks.

---

### 📥 (1) Download the SYNTHIA Dataset

```bash
wget --no-check-certificate http://synthia-dataset.cvc.uab.cat/SYNTHIA_RAND_CITYSCAPES.rar
# (2) For Ubuntu, maybe the rar package is missiong
$sudo apt-get install rar

# (3) unrar the package
$ unrar x SYNTHIA_RAND_CITYSCAPES.rar

# (4) the following directory structure is necessary

synthia
    ├── Depth
    │     └── Depth
    ├── GT
    │     ├── COLOR
    │     └── LABELS
    └── RGB