# Sound-Classification on Dataset-UrbanSound8K

## Overview

This repository contains the implementation of a sound classification system using various machine learning and deep learning techniques. The task involves categorizing audio signals into predefined classes based on their acoustic characteristics. 

### Dataset
We used the **UrbanSound8K** dataset, which contains 8732 labeled sound excerpts (≤4s) of urban sounds categorized into 10 classes:
- `air_conditioner`
- `car_horn`
- `children_playing`
- `dog_bark`
- `drilling`
- `engine_idling`
- `gun_shot`
- `jackhammer`
- `siren`
- `street_music`

Dataset source: [UrbanSound8K](https://huggingface.co/datasets/danavery/urbansound8K)

### Problem Statement
The objective of the project is:
1. **Zero-shot evaluation** of a pre-trained model.
2. Training or fine-tuning models on the domain-specific dataset.
3. Comparing different frameworks based on:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Computational complexity
4. Discussing the advantages and drawbacks of each approach.

---

## Project Structure

- **`MFCC.ipynb`**: Implementation using MFCC features and traditional classifiers.
- **`wav2vec.ipynb`**: Implementation using Wav2Vec2 embeddings for classification.
- **`Report_Augnito_Assignment.pdf`**: Detailed project report including methodology, results, and future directions.
- **`Assignment+-+Sound+Classification.doc`**: Problem statement and initial approach details.

---

## Methodologies

### 1. **MFCC Features with Classifiers**
- Extracted **MFCC (Mel-Frequency Cepstral Coefficients)** to summarize spectral properties of audio signals.
- Trained various classifiers such as:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - KNN
- **Data Augmentation**:
  - Noise addition
  - Pitch shifting

### 2. **Wav2Vec2 Embeddings**
- Leveraged Facebook's pre-trained **Wav2Vec2** model for feature extraction.
- Trained classifiers on these embeddings.

### 3. **Fine-tuning Wav2Vec2**
- Fine-tuned the Wav2Vec2 model with:
  - Additional linear layers for classification
  - Cross-entropy loss
- **LoRA (Low-Rank Adaptation)** was employed to reduce trainable parameters.

---

## Results

| Model                  | Accuracy | Precision | Recall  | F1 Score |
|------------------------|----------|-----------|---------|----------|
| MFCC + XGBoost         | 90.09%   | 90.70%    | 88.92%  | 89.80%   |
| Wav2Vec2 + Logistic Reg| 76.81%   | 76.99%    | 76.28%  | 76.63%   |
| Fine-tuned Wav2Vec2    | 19.51%   | 6.92%     | 16.11%  | 7.52%    |

---

## Tools and Frameworks

- **Programming Language**: Python
- **Libraries**:
  - `Librosa` for audio processing
  - `Scikit-learn` for traditional classifiers
  - `Transformers` for Wav2Vec2 implementation
  - `PyTorch` for deep learning tasks
- **Environment**: Google Colab for training and evaluation.

---

## Future Directions

1. **Advanced Data Augmentation**:
   - Use SpecAugment for spectrogram masking.
   - Add background noises for real-world simulation.
2. **Transfer Learning**:
   - Fine-tune domain-specific Wav2Vec2 models.
   - Use lightweight models like DistilWav2Vec for efficiency.
3. **Balanced Dataset Creation**:
   - Address class imbalance using oversampling or synthetic data generation.
4. **Ensemble Learning**:
   - Combine predictions from CNN, RNN, and Wav2Vec2 models.
5. **Architecture Improvements**:
   - Experiment with hybrid architectures combining CNNs and attention mechanisms.

---

## References
- UrbanSound8K Dataset: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- Pre-trained Model: [Wav2Vec2](https://arxiv.org/abs/2109.15053)

For more details, refer to the **project report** and notebooks in this repository.
