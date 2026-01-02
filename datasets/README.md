# ASL Datasets

This directory stores datasets for training ASL recognition models.

## Directory Structure

```
datasets/
├── kaggle/
│   ├── asl_alphabet/
│   ├── asl_mnist/
│   └── asl_citizen/
├── how2sign/
│   ├── video_clips/
│   ├── annotations/
│   └── README.txt
└── README.md (this file)
```

## Download Datasets

### Kaggle Datasets

```bash
# Automated download
python scripts/download_datasets.py

# Manual download
kaggle datasets download -d grassknoted/asl-alphabet
kaggle datasets download -d datamunge/sign-language-mnist
```

### How2Sign Dataset

1. Request access at: https://how2sign.github.io/
2. Follow download instructions after approval
3. Extract to `datasets/how2sign/`

## Dataset Information

See [docs/DATASETS.md](../docs/DATASETS.md) for:
- Dataset descriptions
- Licensing information
- Citations
- Usage terms

## Dataset Sizes

| Dataset | Size | Files | Format |
|---------|------|-------|--------|
| ASL Alphabet | ~3 GB | 87,000 images | JPG |
| ASL MNIST | ~50 MB | 34,627 images | CSV |
| How2Sign | ~450 GB | 35,000+ videos | MP4 |

## Usage

```python
from src.dataset_loader import DatasetLoader

# Initialize loader
loader = DatasetLoader(config['datasets'])

# Load alphabet dataset
images, labels, classes = loader.load_alphabet_dataset('kaggle')

# Load sentence dataset
videos, sentences = loader.load_sentence_dataset('how2sign')
```

## Data Preparation

Before training:

1. Download datasets
2. Verify integrity
3. Preprocess if needed
4. Split into train/val/test

## Attribution

Always cite the datasets you use:

```
@dataset{asl_alphabet,
  title={ASL Alphabet},
  author={Kaggle Community},
  year={2018},
  url={https://www.kaggle.com/datasets/grassknoted/asl-alphabet}
}
```

See [docs/DATASETS.md](../docs/DATASETS.md) for full citations.

## Privacy and Ethics

- Respect participant privacy
- Follow dataset usage terms
- Be aware of demographic biases
- Consider cultural context of ASL

---

**Note**: Datasets are NOT included in this repository due to size and licensing. You must download them separately.
