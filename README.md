
# Customer Churn Prediction using Artificial Neural Network

This project applies a deep learning approach using an Artificial Neural Network (ANN) built with TensorFlow/Keras to predict whether a bank customer is likely to churn (i.e., leave the bank).

---

## Dataset

- **File:** `Churn_Modelling.csv`
- **Source Columns:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Target:** `Exited` (1 = churned, 0 = stayed)

---

## Workflow

### 1. **Data Preprocessing**
- Load dataset using `pandas`.
- Extract features (`x`) and target (`y`).
- Convert categorical variables (`Geography`, `Gender`) using `pd.get_dummies`.
- Scale features using `StandardScaler`.
- Train-test split (80/20).

### 2. **ANN Architecture**
- Framework: `TensorFlow` with `Keras Sequential API`
- **Layers:**
  - Input layer with 11 neurons (ReLU) + Dropout(0.2)
  - Hidden layer 1: 7 neurons (ReLU) + Dropout(0.3)
  - Hidden layer 2: 6 neurons (ReLU)
  - Output layer: 1 neuron (Sigmoid)

### 3. **Compilation & Training**
- Optimizer: `Adam`
- Loss function: `binary_crossentropy`
- Metric: `accuracy`
- Epochs: 100, Batch size: 10

### 4. **Model Evaluation**
- Prediction on test set.
- Performance metrics:
  - Confusion Matrix
  - Accuracy Score

---

## Sample Results

- Trained model achieves high accuracy on unseen data.
- ANN effectively identifies customers likely to churn.

---

## Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

## How to Run

```bash
# Install required packages
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook ANN.ipynb
```

---

## Notes

- Dropout regularization is used to prevent overfitting.
- Model structure can be tuned further using different activation functions (e.g., ELU, PReLU).
- Consider saving the trained model for deployment (e.g., `model.save()`).
