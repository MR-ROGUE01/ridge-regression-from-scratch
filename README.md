# 📈 Ridge Regression From Scratch

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-Matrix%20Operations-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Comparison-green)

A complete implementation of **Ridge Regression from Scratch** using:

✔️ Closed Form Solution  
✔️ Gradient Descent  
✔️ Single Feature Regression  
✔️ Multiple Feature Regression  
✔️ Comparison with Scikit-Learn

---

## 🧠 What is Ridge Regression?

Ridge Regression is an extension of Linear Regression that adds **L2 Regularization** to reduce overfitting.

### Cost Function

\[
J(\theta)=||y-X\theta||^2+\alpha||\theta||^2
\]

Where:

```text
α = Regularization Strength
θ = Model Parameters
X = Feature Matrix
y = Target Variable
```

---

## 🚀 Closed Form Solution

```python
I = np.identity(X.shape[1])
I[0][0] = 0

theta = np.linalg.inv(
    X.T @ X + alpha * I
) @ X.T @ y
```

---

## ⚡ Gradient Descent Implementation

```python
reg_term = alpha * theta
reg_term[0] = 0

gradient = (
    X.T @ X @ theta
    - X.T @ y
    + reg_term
)

theta = theta - learning_rate * gradient
```

---

## 📊 Visualization

Different values of α shrink coefficients differently.

```python
alpha = 1
alpha = 10
alpha = 100
```

Result:

```text
Higher Alpha
      ↓
Smaller Coefficients
      ↓
Lower Variance
      ↓
Less Overfitting
```

---

## 📂 Project Structure

```text
ridge-regression-from-scratch/
│
├── ridge_regression.ipynb
├── README.md
└── requirements.txt
```

---

## 🔍 Comparison With Scikit-Learn

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train,y_train)
```

### Custom Implementation

```python
reg = MeraRidge(alpha=1.0)
reg.fit(X_train,y_train)
```

Both produce nearly identical coefficients and R² scores.

---

## 🛠️ Tech Stack

```text
Python
NumPy
Matplotlib
Scikit-Learn
Jupyter Notebook
```

---

## 🎯 Learning Outcomes

✅ Ridge Regression Mathematics

✅ L2 Regularization

✅ Matrix Algebra

✅ Gradient Descent

✅ Hyperparameter Tuning

✅ Model Comparison

---

## 👨‍💻 Author

Raj (MR-ROGUE01)

B.Tech CSE (AI & ML)

GitHub:
https://github.com/MR-ROGUE01
