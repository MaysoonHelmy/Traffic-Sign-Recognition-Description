# Traffic Sign Recognition – CNN vs MobileNetV2

##  Description
This project implements a **Traffic Sign Recognition system** using the **GTSRB (German Traffic Sign Recognition Benchmark) dataset**.  
It is part of the **Self-Paced Elevvo Machine Learning Internship (Industry-Level Project)**.  

The notebook compares two approaches for classifying traffic signs:  
1. **Custom Convolutional Neural Network (CNN)** – built from scratch.  
2. **MobileNetV2 (Pretrained Model)** – fine-tuned for traffic sign classification.  

---

##  Features
- Image preprocessing and normalization  
- Data augmentation for better generalization  
- Two model approaches:
  - Custom CNN  
  - MobileNetV2 (transfer learning)  
- Evaluation using accuracy, precision, recall, and F1-score  
- Visual performance comparison between models  

---

##  Project Workflow
1. **Dataset Loading and Analysis**  
   - [GTSRB dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)  
   - Images resized and normalized  

2. **Data Preprocessing & Augmentation**  
   - Applied transformations (rotation, zoom, shifts)  

3. **Model Development**  
   - Custom CNN with convolution, pooling, dropout, and dense layers  
   - MobileNetV2 pretrained on ImageNet and fine-tuned  

4. **Training & Validation**  
   - Trained both models with early stopping and validation tracking  

5. **Evaluation**  
   - Compared class-wise **precision, recall, and F1-score**  

## ⚙️ Installation
Install the dependencies with:  

```bash
pip install tensorflow keras matplotlib seaborn scikit-learn opencv-python
````

Dataset: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

---

## 🚀 Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```
2. Open the notebook:

   ```bash
   jupyter notebook traffic_sign_recognition.ipynb
   ```
3. Run all cells to train and evaluate both models.

---

## 🏷️ Tags & Features

* **Tags:** Deep Learning, CNN, Transfer Learning, MobileNetV2, Image Classification, Traffic Signs, Computer Vision, Internship Project
* **File Types:** `.ipynb` (Jupyter Notebook), dataset images (`.png`)

---

## 📊 Output Features

The notebook outputs:

* Training vs Validation vs Test Accuracy & Loss plots
* Class-wise Precision, Recall, and F1-score plots

### Test Accuracy comparison between 2 models
    This shows how the pretrained model generalizes better
    
  <img width="586" height="473" alt="image" src="https://github.com/user-attachments/assets/bd073214-37d8-43e4-9ba2-71c0068ae60f" />
    
### Class-wise F1-score Comparison
    This plot compares the performance of the CNN and MobileNetV2 across different traffic sign classes.
    
   <img width="847" height="484" alt="image" src="https://github.com/user-attachments/assets/9ec79e95-da41-4dea-a49a-b1bdc526855f" />

## ✅ Conclusion

* The **Custom CNN** performed well overall but showed weaknesses in some classes, especially rare or complex signs.
* The **MobileNetV2 model** consistently outperformed the custom CNN with **higher accuracy, precision, recall, and F1-scores**.
* This shows that **transfer learning with pretrained models is more effective** for real-world problems like traffic sign recognition.
* In practical applications such as **autonomous driving**, this higher accuracy can directly improve safety and reliability.

