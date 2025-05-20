
# TEAD: Trust-Enhanced Anomaly Detection Framework for IoT-Enabled WSNs

This repository contains the implementation of the TEAD framework, a hybrid model for intrusion detection in IoT-enabled Wireless Sensor Networks. It integrates dynamic trust evaluation, anomaly detection via KL divergence, and deep learning classification using LSTM.

---

## 📂 Project Structure

- `TEAD_Part2_KL_AnomalyDetection.py`  
  Computes trust scores (CLR, CWR, CFD) and applies Gaussian modeling with KL divergence to flag anomalies.

- `TEAD_Part3_LSTM_Classification.py`  
  Uses LSTM for node behavior classification based on trust metrics.

- `TEAD_Part4_Evaluation_Visualization.py`  
  Evaluates model performance using standard metrics and visualizes confusion matrix and accuracy trends.

---

## 📊 Dataset

The implementation uses the **NSL-KDD** dataset.

- You can load it directly from:  
  `https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt`

---

## 🚀 Running in Google Colab

1. Upload all `.py` files or copy the code cells into a Colab notebook.
2. Run Part 2 to compute trust and detect anomalies.
3. Run Part 3 to train and evaluate the LSTM classifier.
4. Run Part 4 to visualize performance.

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- KL Divergence values
- Trust Score Distribution

---

## 📜 Citation

If you use this code or framework in your research, please cite the corresponding paper:

> **[Your Paper Title]**  
> Authors: [Your Names]  
> Submission: [Conference/Journal Name]  
> DOI: [Add upon publication]

---

## 📬 Contact

For questions, please contact [rajawaseem@gmail.com] or open an issue.
