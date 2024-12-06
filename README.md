# House Price Prediction Project

## **Overview**
This project aims to predict house prices using machine learning models. By analyzing various features of the houses and their respective sale prices, we built and fine-tuned predictive models to achieve high accuracy. The project was structured into three main steps: **Exploratory Data Analysis (EDA)**, **Model Selection**, and **Model Fine-Tuning**. 

Our primary focus was to build models that minimize error while providing valuable insights into feature importance. The project uses data preprocessing, feature selection techniques, and advanced machine learning algorithms.

---

## **Project Structure**
1. **Exploratory Data Analysis (EDA)**:
   - Performed data cleaning and feature engineering.
   - Split the data into training and testing datasets.
   - Saved the processed datasets as CSV files for later use.

2. **Model Selection**:
   - Scaled the data and trained several machine learning models:
     - **Linear Regression**
     - **Support Vector Regression (SVR)**
     - **Random Forest Regressor**
     - **XGBoost Regressor**
   - Evaluated model performance using **RMSE**, **MAE**, and **\(R^2\)** metrics.
   - Implemented **Recursive Feature Elimination (RFE)** and other feature selection techniques to simplify the models without sacrificing performance.

3. **Model Fine-Tuning**:
   - Fine-tuned the best models (**Random Forest** and **XGBoost**) using **GridSearchCV** and **RandomizedSearchCV**.
   - Combined the models using **Stacking Regressor** to leverage the strengths of each model.
   - Evaluated the final model on unseen test data.

---

## **Models and Results**
We evaluated the following models and compared their performance:

| **Model**                           | **Train RMSE ($)** | **Test RMSE ($)** | **Train MAE ($)** | **Test MAE ($)** | **Train \(R^2\)** | **Test \(R^2\)** |
|-------------------------            |--------------------|-------------------|-------------------|------------------|-------------------|------------------|
| **Linear Regression**               | 218,190.02         | 226,027.37        | 139,374.83        | 144,462.21       | 0.43              | 0.47             |
| **Support Vector Regression (SVR)** | 53,258.56          | 78,812.54         | 31,174.77         | 37,704.67        | 0.97              | 0.94             |
| **Random Forest**                   | 13,620.01          | 34,545.65         | 4,470.46          | 10,315.65        | 1.00              | 0.99             |
| **XGBoost**                         | 4,245.25           | 21,291.41         | 2,660.82          | 5,472.93         | 1.00              | 1.00             |
| **Stacking Regressor**              | 4,459.16           | 21,235.93         | 3,010.57          | 5,673.68         | 1.00              | 1.00             |

---

## **Feature Selection**
### **Techniques Used**:
1. **Recursive Feature Elimination (RFE)**:
   - Simplified the dataset by iteratively removing less important features.
   - Retained top-performing features for each model.

2. **Feature Importances**:
   - Leveraged the feature importance scores from **Random Forest** and **XGBoost**.
   - Selected features that contributed the most to model performance.

### **Key Features Identified**:
- `description_baths`
- `description_sqft`
- `fireplace`
- `view`
- `community_security_features`
- `state_frequency`
- `days_on_market`

---

## **Challenges**
1. **Dataset Limitations**:
   - The dataset size and timeframe were limited, which might have constrained the models' ability to generalize to broader market conditions.
   - Missing values in certain features required significant preprocessing, which may have reduced the richness of the data.

2. **Model Complexity**:
   - Advanced models like **XGBoost** and **Stacking Regressors** require careful hyperparameter tuning, which was time-intensive.

3. **Interpretability**:
   - Complex models, while highly accurate, are more difficult to interpret, which could pose challenges for real-world adoption.

---

## **Future Goals**
1. **Fine-Tune Further**:
   - Experiment with additional hyperparameter combinations for Random Forest and XGBoost to push performance further.
   - Include ensemble techniques like **LightGBM** and **CatBoost** for comparison.

2. **Feature Engineering**:
   - Create new features based on domain knowledge (e.g., neighborhood-level data, economic trends).
   - Incorporate temporal data to capture market changes over time.

3. **Expand Dataset**:
   - Use a larger dataset spanning multiple regions and years to make the model more robust and generalizable.

4. **Model Deployment**:
   - Deploy the final model via an API for real-time predictions.
   - Build a user-friendly dashboard for non-technical users to input house details and receive predictions.

5. **Enhance Explainability**:
   - Use tools like **SHAP (SHapley Additive exPlanations)** to improve model transparency and highlight the impact of individual features.

5. **Code Enhancement**:
   - U
---

## **Final Model**
The **Stacking Regressor** combining **Random Forest** and **XGBoost** was the best-performing model. It achieved:

- **Test RMSE**: $24254.97
- **Test MAE**: $4858.01
- **Test \(R^2\)**: 0.99

This model demonstrated excellent performance, with minimal error and perfect generalization to unseen data.

---

## **Tools and Libraries**
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Machine Learning: `scikit-learn`, `xgboost`
  - Visualization: `matplotlib`, `seaborn`
- **Environment**: Jupyter Notebooks

---



