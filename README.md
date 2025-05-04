# NeuroNexus
# Titanic Survival Prediction using Machine Learning

This project aims to build a machine learning model that predicts whether a passenger on the Titanic survived or not, based on various attributes such as age, sex, class, and fare. The dataset used is the classic Titanic dataset provided by [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

---

## 📌 Objective

To develop a supervised classification model using machine learning techniques to predict passenger survival based on demographic and socio-economic data available in the Titanic dataset.

---

## 📂 Dataset Source

- **Title**: Titanic - Machine Learning from Disaster  
- **URL**: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)  
- **Files Used**: `train.csv`, `test.csv`

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Kaggle API

---

## 🔍 Project Workflow

1. **Dataset Access via Kaggle API**  
   Automatically downloads Titanic dataset using your Kaggle credentials.

2. **Data Preprocessing**
   - Handling missing values (Age, Fare, Embarked)
   - Dropping non-informative columns (`Cabin`, `Name`, `Ticket`)
   - Encoding categorical features (`Sex`, `Embarked`)

3. **Model Building**
   - Trained using `RandomForestClassifier`
   - Data split into training and testing sets (80/20)
   - Evaluated using accuracy, confusion matrix, and classification report

4. **Prediction**
   - Survival prediction made on `test.csv`
   - Results displayed per passenger and saved as `submission.csv`

5. **Visualization**
   - A survival count bar chart with labeled integer values
   - Feature importance bar plot

---

## 📈 Obtained Output
plt.savefig("survival_count_plot.png")
