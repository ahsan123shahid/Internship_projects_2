# Credit Card Fraud Detection

This project implements a **Credit Card Fraud Detection System** using **RandomForestClassifier**. It processes transactions from the `creditcard.csv` dataset, balances the data using undersampling, and predicts fraudulent transactions with high accuracy.

## 📌 Features
- Uses **RandomForestClassifier** for fraud detection
- **Balances imbalanced data** using undersampling
- **Scales transaction amount** using StandardScaler
- **Predicts new transactions** as fraud or non-fraud
- **Optimized for speed** while maintaining high accuracy

## 🚀 Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install pandas scikit-learn imbalanced-learn
```

### 3. Download the Dataset
The dataset (`creditcard.csv`) is available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). Place it in the project folder.

## 🏗 Usage
### 1. Run the Fraud Detection Model
```bash
python fraud_detection.py
```

### 2. Predict a New Transaction
Modify the `new_transaction` dictionary in the script and re-run it.

## 📊 Model Performance
- **Precision:** 96.15%
- **Recall:** 92.59%
- **F1 Score:** 94.3%

## 🛠 Future Improvements
- Implement **XGBoost** for better accuracy
- Convert the model into a **Flask API** for real-time predictions
- Experiment with **AutoML (TPOT, H2O.ai)**

## 🤝 Contributing
Feel free to submit pull requests or report issues!

## 📜 License
This project is licensed under the MIT License.

---
⭐ If you found this helpful, give it a star on GitHub! ⭐

