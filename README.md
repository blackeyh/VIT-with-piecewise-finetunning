# 😄 **Facial Expression Classification Using Vision Transformers with Piecewise Fine-Tuning**

Welcome to an exciting deep learning project that leverages the power of **Vision Transformers (ViT)** to classify facial expressions with **stunning accuracy**! This project uses an innovative and **high-performance** strategy known as **Piecewise Fine-Tuning** to train a facial emotion classifier, breaking new ground in model performance and generalization.

We aim to classify facial expressions into four key categories: **Angry**, **Normal**, **Tongue Out**, and **Happy**. This project goes above and beyond by evaluating **ViT** against state-of-the-art CNN models like **VGG16**, **InceptionV3**, **DenseNet121**, **MobileNet**, **NASNet**, and even other **Vision Transformer models**.

What makes this project truly remarkable is the **Piecewise Fine-Tuning Strategy**, which optimizes the learning process in two distinct phases, achieving **unmatched generalization**.

---

## 📂 **Dataset Overview**

The dataset used in this project contains facial images, each representing one of four emotional expressions:

- 😡 **Angry**
- 😐 **Normal**
- 😜 **Tongue Out**
- 😁 **Happy**

The diversity of facial expressions allows us to explore how well the model can generalize to different emotional states.

---

## 🔄 **Piecewise Fine-Tuning Strategy**

The secret to the superior performance of our model lies in the **Piecewise Fine-Tuning Strategy**. This two-phase training approach ensures the model not only learns to generalize better but also specializes on the finer details of facial cues. Here’s how it works:

### 🔹 **Phase 1: Pretraining on Augmented Data**
In this phase, the model is trained exclusively on **brightness-augmented images**. These augmented images simulate real-world lighting variations and challenge the model to **generalize across lighting conditions**. This phase allows the model to learn **low-level features** without being distracted by specific image artifacts.

### 🔹 **Phase 2: Fine-Tuning on Clean Data**
Once the model has learned to generalize, it enters the **fine-tuning phase**. Here, the model is exposed to clean, original images of the facial expressions. This enables it to specialize in **fine-grained facial features** and **emotional cues** without overfitting to noise or lighting artifacts.

By carefully dividing training into these two phases, the model strikes the perfect balance between robustness and specialization.

---

## 🧠 **Model Architecture & Components**

The backbone of our model is the **Vision Transformer (ViT)**, a state-of-the-art model known for its exceptional performance in image classification tasks. Here's an overview of the architecture and key components:

| **Component**        | **Details**                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| **Backbone Model**   | `ViTForImageClassification` (from HuggingFace Transformers)                  |
| **Pretrained Model** | `dima806/facial_emotions_image_detection`                                    |
| **Image Processor**  | `ViTImageProcessor` — performs resizing, normalization, and tokenization     |
| **Input Size**       | 224x224 RGB                                                                  |
| **Labels**           | {'angry': 0, 'normal': 1, 'tongue': 2, 'happy': 3}                           |
| **Loss Function**    | **Cross-Entropy Loss** — optimal for multi-class classification              |

### 🔍 **Why Cross-Entropy Loss?**
Cross-Entropy Loss is the ideal choice for multi-class classification problems like ours because it **penalizes incorrect predictions** based on the model's confidence, ensuring that the model's accuracy improves as it learns.

---

## 🔧 **Preprocessing Pipeline**

To prepare the data for training, we employ a sophisticated preprocessing pipeline:

- **Image Conversion**: Convert images to RGB format using **PIL**.
- **ViTImageProcessor**:
  - Resize images to **224x224**
  - Normalize images using **ImageNet statistics** for better convergence
  - Convert images into PyTorch **tensors**
- **Efficient Storage**: The processed tensors are stored as `.pt` files to optimize memory usage during training.

### 🗂 **DiskImageDataset**
A custom **PyTorch Dataset** is used to load the preprocessed tensors from disk. It returns the **pixel values** and the corresponding **labels**, enabling efficient data loading during training.

---

## ⚡ **Training Details**

Training the model efficiently is a crucial step in ensuring its high performance. Here are the key hyperparameters and training settings:

| **Hyperparameter**    | **Value**                            |
| --------------------- | ------------------------------------ |
| **Batch Size**        | 32                                   |
| **Optimizer**         | **AdamW** (adaptive learning rate)  |
| **Weight Decay**      | Enabled (default for regularization) |
| **Loss Function**     | **Cross-Entropy Loss**              |
| **Train/Test Split**  | 80% / 20%                            |
| **Learning Rate**     | HuggingFace default                  |

### 🗂 **Model Saving Logic**
We use a custom callback, **CustomSaveModelCallback**, to save the model at each evaluation point. The saved models are named with the **epoch number** and **evaluation accuracy**, for example: `model_epoch_2.00_acc_0.9235.bin`.

---

## 📊 **Evaluation Metrics**

The **confusion matrix** shown below highlights the model's remarkable performance on validation data. The results speak for themselves, demonstrating that the **ViT model outperforms all traditional CNN architectures** in terms of accuracy and precision.

![Confusion Matrix](confusion_matrix.png)

Additionally, the comparison table shows that our ViT-based model achieves **significantly better results** compared to popular CNN architectures such as **VGG16**, **InceptionV3**, and others.

![Model Comparison](comparison_table.png)

---

## 📚 **Libraries Used**

To implement this powerful facial emotion classification model, we used the following libraries and tools:

- **PyTorch** & **Torchvision** (for deep learning)
- **HuggingFace Transformers** (for ViT and pre-trained models)
- **PIL** (for image processing)
- **scikit-learn** (for evaluation and metrics)
- **pandas**, **os**, **glob** (for data handling and file operations)

---

## 🔍 **Summary**

In this project, we introduced a **Piecewise Fine-Tuning Strategy** to optimize **Vision Transformers (ViT)** for facial emotion recognition. The model begins with a pretraining phase using **augmented data**, allowing it to generalize to diverse lighting conditions, and then fine-tunes on clean data for specialization in facial features. This two-phase approach allows the model to **achieve excellent performance** with superior generalization.

By employing this innovative training strategy, we demonstrate that **ViT-based models can outperform traditional CNN architectures** on facial expression classification tasks.

---

## 🚀 **Next Steps**

- **Model Improvement**: Experiment with additional augmentation techniques and regularization methods to further improve generalization.
- **Real-Time Application**: Integrate the model into real-time facial emotion detection systems for practical applications like **virtual assistants**, **gaming**, and **user experience research**.
- **Multi-Class Extension**: Explore extending the model to handle more emotional states or facial actions.

---

## 💬 **Contribute**

Feel free to contribute to this project by submitting pull requests, reporting issues, or suggesting improvements. Together, we can make this facial emotion classifier even better!

---

## ✨ **Acknowledgments**

Thanks to the **HuggingFace Transformers** team for providing the ViT architecture and pre-trained models that made this project possible. Special thanks to the **PyTorch** and **scikit-learn** communities for their amazing tools and resources.

---

🚀 **Let's bring facial emotion classification to the next level with Vision Transformers!**
