![Disease Prediction Banner](assets/dp_banner_update.png)

# 🩺 Disease Prediction Model  
### *Machine Learning Project Assessment (Week 3)*

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Project-Machine%20Learning-success)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare-critical)
![Status](https://img.shields.io/badge/Status-Academic%20Portfolio%20Project-informational)

**Author:** Bernadette Burks  
**Date:** September 14, 2025  

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Required Technologies](#-required-technologies)
- [Python Libraries](#-python-libraries)
- [Dataset Details](#-dataset-details)
- [Data Preparation & Cleaning](#-data-preparation--cleaning)
- [Model Training Methods](#-model-training-methods)
- [Model Evaluation](#-model-evaluation)
- [Example Prediction Workflow](#-example-prediction-workflow)
- [Key Skills Demonstrated](#-key-skills-demonstrated)
- [Future Improvements](#-future-improvements)
- [Project Structure](#-project-structure)
- [References](#-references)

---

## 📌 Project Overview

For this assignment, I developed a **Disease Prediction Model** based on the GeeksforGeeks tutorial:  
https://www.geeksforgeeks.org/machine-learning/disease-prediction-using-machine-learning/

This project was selected intentionally to align with my ongoing healthcare-related academic work.

The primary goal of this model is to explore how machine learning can assist in identifying diseases based on symptom patterns commonly presented in clinical environments, such as Emergency Departments.

---

## ⚙️ Required Technologies

One of the strengths of this project is its accessibility: it requires only a Python-compatible IDE with current updates applied, making it approachable for beginners entering the field of machine learning.

For development, I used **VS Code (Python 3.13)** as this is a frequently used IDE in enterprise settings.

---

## 📚 Python Libraries

This project incorporates several foundational libraries widely used in data science workflows:

- **Pandas** – Provides efficient data structures and visualization support built on NumPy.  
- **NumPy** – Enables fast numerical operations and compact dataset handling.  
- **SciPy** – Extends NumPy with additional scientific computing functionality.  
- **Matplotlib** – Offers highly customizable plotting tools compatible with most ML libraries.  
- **Seaborn** – Builds upon Matplotlib for more complex statistical visualizations.  
- **Scikit-learn** – Supplies integrated machine learning algorithms and model evaluation tools.

---

## 🗂 Dataset Details

The dataset used in this project is entitled **improved_disease_dataset** and contains a single file with **2,000 total entries**.

### Features Included

The dataset consists of symptom-based predictor variables, including:

- fever  
- headache  
- nausea  
- vomiting  
- fatigue  
- joint_pain  
- skin_rash  
- cough  
- weight_loss  
- yellow_eyes  

The outcome variable is:

- **disease** (the predicted diagnosis)

---

### 📊 Training & Validation Approach

This dataset is cross-validated using **5-fold stratified k-fold validation**, which defaults to:

- **80% training data (4 folds)**
- **20% validation/testing data (1 fold)**

This results in:

- Training set: **1,600 rows**
- Validation set: **400 rows**

---

## 🧹 Data Preparation & Cleaning

A recommended preprocessing step involves converting disease labels into numeric values to support early visualization and detection of class imbalance.

Because certain disease categories were underrepresented, the dataset benefits from applying **RandomOverSampler**, ensuring that each disease class receives equal representation during training.

This balancing improves the model’s ability to generalize rather than overfitting to dominant categories.

---

## 🧠 Model Training Methods

For cross-validation, the project evaluates three primary machine learning classifiers:

- `DecisionTreeClassifier()`  
- `RandomForestClassifier()`  
- `SVC()`  

However, during the confusion matrix evaluation stage, the `DecisionTreeClassifier()` appears to be replaced with `GaussianNB()`.

After researching this discrepancy and finding limited discussion online, I consulted ChatGPT for clarification.

---

### 💬 ChatGPT Clarification

> This looks like a design inconsistency, not an intentional strategy:  
>
> • The CV section should ideally include all models you care about evaluating, to compare them fairly.  
> • Adding Naive Bayes only later (without CV scores) means you don’t know how it performs under cross-validation — only how it memorizes the whole dataset (which can inflate accuracy).  
>
> So yes — I’d call this at least a mistake in consistency. The author probably meant to include GaussianNB in the CV loop but forgot to add it to models.

---

### ✅ Best Practice Recommendation

- Include all candidate models in the cross-validation loop  
  (Decision Tree, Random Forest, SVM, Naive Bayes).  
- Retrain afterward only for confusion matrices or final ensemble predictions.

---

## 📈 Model Evaluation

The project concludes by producing a final predictive function, **predict_disease**, which accepts symptom inputs and outputs a predicted diagnosis.

Across the trained models, two out of three classifiers (67%) produced the same disease prediction given identical symptom sets.

While this performance level would require significant improvement before clinical deployment, it serves as a strong educational foundation and demonstrates the potential for future refinement in healthcare-oriented machine learning applications.

---

## 🧪 Example Prediction Workflow

Once trained, the model can be used by inputting symptom features such as:

```python
predict_disease(
    fever=1,
    headache=1,
    nausea=0,
    vomiting=0,
    fatigue=1,
    joint_pain=0,
    skin_rash=0,
    cough=1,
    weight_loss=0,
    yellow_eyes=0
)
```
---

## 🛠 Key Skills Demonstrated

This project highlights several core machine learning and healthcare analytics skills:

- Data preprocessing and label encoding
- Handling class imbalance with oversampling
- Model training with cross-validation
- Comparing classifier performance
- Confusion matrix evaluation
- Applying ML concepts in clinical prediction contexts

---

## 🚀 Future Improvements

This project serves as an excellent starting point for continued development. Future enhancements may include:

- Expanding model evaluation metrics beyond accuracy (precision, recall, F1-score)
- Including all candidate models consistently in cross-validation
- Testing additional algorithms such as Gradient Boosting or XGBoost
- Improving interpretability using feature importance tools
- Exploring real-world clinical datasets for stronger generalization

---

## 🔗 References

ChatGPT. (n.d.). Retrieved September 14, 2025 from [https://chatgpt.com/](https://chatgpt.com/)

Disease Prediction Using Machine Learning. (2025). GeeksforGeeks. Retrieved September 14, 2025 from [https://www.geeksforgeeks.org/machine-learning/disease-prediction-using-machine-learning/](https://www.geeksforgeeks.org/machine-learning/disease-prediction-using-machine-learning/)

Ly, S. (2024). 8 Python Libraries You Must Know for Data Science. Simple Analytics. Retrieved September 14, 2025 from [https://simpleanalytics.co.nz/blogs/8-python-libraries-you-must-know-for-data-science](https://simpleanalytics.co.nz/blogs/8-python-libraries-you-must-know-for-data-science)

SciPy. (n.d.). Retrieved September 14, 2025 from [https://scipy.org/](https://scipy.org/)
