# Dataset Information and Attributions

This document provides information about the datasets used in the ASL Vision Bot project, including licensing, citations, and download instructions.

## Datasets

### 1. Kaggle ASL Datasets

#### ASL Alphabet Dataset
- **Source**: [Kaggle - ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Description**: Images of American Sign Language alphabet signs (A-Z, space, delete, nothing)
- **Size**: ~87,000 images
- **Format**: JPG images (200x200)
- **License**: Varies - check Kaggle dataset page for specific license
- **Citation**: 
  ```
  ASL Alphabet Dataset, Kaggle
  https://www.kaggle.com/datasets/grassknoted/asl-alphabet
  ```

#### ASL MNIST Dataset
- **Source**: [Kaggle - Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Description**: MNIST-like dataset for ASL alphabet (A-Z excluding J and Z which require motion)
- **Size**: 27,455 training images, 7,172 test images
- **Format**: CSV files with pixel values (28x28 grayscale)
- **License**: CC0: Public Domain
- **Citation**:
  ```
  Sign Language MNIST, Kaggle
  https://www.kaggle.com/datasets/datamunge/sign-language-mnist
  ```

#### ASL Citizen Dataset (if used)
- **Source**: Various Kaggle datasets
- **Description**: Diverse collection of ASL signs from multiple contributors
- **License**: Varies by dataset
- **Note**: Check individual dataset licenses on Kaggle

### 2. How2Sign Dataset

- **Source**: [How2Sign Official Website](https://how2sign.github.io/)
- **Description**: Large-scale multimodal dataset for continuous American Sign Language
- **Size**: ~35,000 sentence instances, ~16,000 unique sentences
- **Format**: Video clips (MP4) with annotations (JSON)
- **License**: Research and Educational Use Only
- **Citation**:
  ```
  @inproceedings{duarte2021how2sign,
    title={How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language},
    author={Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and 
            DeHaan, Kenneth and Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={2735--2744},
    year={2021}
  }
  ```

### 3. MediaPipe Hand Landmarker

- **Source**: [Google MediaPipe](https://developers.google.com/mediapipe)
- **Description**: Pre-trained model for hand landmark detection
- **License**: Apache License 2.0
- **Citation**:
  ```
  @misc{mediapipe,
    title={MediaPipe Hands: On-device Real-time Hand Tracking},
    author={Zhang, Fan and Bazarevsky, Valentin and Vakunov, Andrey and Tkachenka, Andrei and 
            Sung, George and Chang, Chuo-Ling and Grundmann, Matthias},
    year={2020},
    howpublished={\url{https://arxiv.org/abs/2006.10214}}
  }
  ```

## Dataset Download Instructions

### Kaggle Datasets

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Set up API credentials**:
   - Go to Kaggle Account settings: https://www.kaggle.com/account
   - Click "Create New API Token"
   - Place the downloaded `kaggle.json` in `~/.kaggle/`
   - On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download datasets**:
   ```bash
   python scripts/download_datasets.py
   ```
   
   Or manually:
   ```bash
   kaggle datasets download -d grassknoted/asl-alphabet -p datasets/kaggle/asl_alphabet --unzip
   kaggle datasets download -d datamunge/sign-language-mnist -p datasets/kaggle/asl_mnist --unzip
   ```

### How2Sign Dataset

1. **Request Access**:
   - Visit: https://how2sign.github.io/
   - Read the terms of use carefully
   - Fill out the data request form
   - Wait for approval (may take several days)

2. **Download**:
   - Follow the download instructions provided after approval
   - The dataset is large (~450GB for full video dataset)
   - Consider downloading only the subset you need

3. **Extract**:
   ```bash
   # Extract to the datasets directory
   unzip how2sign.zip -d datasets/how2sign/
   ```

4. **Verify structure**:
   ```
   datasets/how2sign/
   ├── video_clips/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── annotations/
   │   ├── train_annotations.json
   │   ├── val_annotations.json
   │   └── test_annotations.json
   └── README.txt
   ```

## Usage Terms and Conditions

### General Guidelines

1. **Academic and Research Use**: All datasets can be used for academic research
2. **Commercial Use**: Check individual dataset licenses for commercial use restrictions
3. **Attribution**: Always cite the original datasets in publications
4. **Redistribution**: Do not redistribute datasets without permission from original authors
5. **Privacy**: Respect privacy of individuals in the datasets

### Specific Restrictions

- **How2Sign**: Research and educational use only. Cannot be used for commercial purposes without explicit permission
- **Kaggle Datasets**: Follow individual dataset licenses on Kaggle
- **MediaPipe Models**: Apache 2.0 - permissive for commercial use

## Data Ethics and Privacy

This project is committed to ethical use of data:

1. **Informed Consent**: All datasets should come from sources where participants gave informed consent
2. **Cultural Sensitivity**: ASL is a living language with rich cultural context. Respect the Deaf community
3. **Bias Awareness**: Be aware of potential biases in datasets (demographics, regional variations)
4. **Accessibility**: Consider how this technology benefits the Deaf community

## Contributing New Datasets

If you want to contribute or suggest new datasets:

1. Ensure proper licensing and permissions
2. Verify data quality and annotations
3. Provide clear documentation
4. Include attribution information
5. Consider ethical implications

## Contact

For questions about dataset usage or licensing:
- Check the original dataset sources
- Open an issue on GitHub
- Contact the dataset creators directly

## Acknowledgments

We acknowledge and thank:
- The Deaf community for their contributions to ASL datasets
- Kaggle contributors who created and shared ASL datasets
- The How2Sign research team at Carnegie Mellon University and UPC Barcelona
- Google MediaPipe team for hand tracking models
- All dataset annotators and volunteers

---

**Last Updated**: 2024
**Version**: 1.0
