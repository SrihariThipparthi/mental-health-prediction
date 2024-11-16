# Mental Health Prediction: Kaggle Playground Competition

## **Overview**
This project aims to predict whether an individual is experiencing depression based on various personal and environmental factors. The dataset used is from a Kaggle Playground competition. The goal is to analyze and model the data to determine the presence or absence of depression effectively.

**Competition Data:** [Kaggle Playground Series S4E11](https://www.kaggle.com/competitions/playground-series-s4e11/data)  

---

## **Dataset**
The dataset contains the following features:

| **Column Name**                       | **Description**                                   |
|---------------------------------------|---------------------------------------------------|
| `id`                                  | Unique identifier for each individual            |
| `Gender`                              | Gender of the individual                         |
| `Age`                                 | Age of the individual                            |
| `Working Professional or Student`     | Work/education status                            |
| `Academic Pressure`                   | Level of academic pressure                       |
| `Work Pressure`                       | Level of work pressure                           |
| `CGPA`                                | Cumulative Grade Point Average (for students)    |
| `Study Satisfaction`                  | Satisfaction level with academic studies         |
| `Job Satisfaction`                    | Satisfaction level with job                      |
| `Sleep Duration`                      | Average sleep duration per day                   |
| `Dietary Habits`                      | Eating habits                                    |
| `Have you ever had suicidal thoughts?`| History of suicidal thoughts                     |
| `Work/Study Hours`                    | Hours spent working or studying daily            |
| `Financial Stress`                    | Level of financial stress                        |
| `Family History of Mental Illness`    | Whether mental illness runs in the family        |
| `Depression`                          | Target variable (1 = Depressed, 0 = Not Depressed) |

---

## **Technologies and Tools Used**
### **Technologies:**
1. **Data Cleaning**: Removing inconsistencies and handling missing values.
2. **Data Manipulation**: Transforming and structuring data for analysis.
3. **Feature Engineering**: Creating new features to enhance model performance.
4. **Exploratory Data Analysis (EDA)**: Visualizing and analyzing patterns in the data.
5. **Machine Learning**: Developing supervised learning models for prediction.
6. **Feature Selection**: Identifying and retaining the most impactful features.
7. **Model Deployment**: Deploying the best-performing model for predictions.

### **Tools:**
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Building machine learning models.
- **Matplotlib**: Data visualization.
- **Seaborn**: Statistical data visualization.
- **LightGBM**: Gradient boosting for classification tasks.
- **Warnings**: Suppressing unnecessary warnings during execution.

---

## **Feature Engineering**
To improve model performance, new features were created:
- **Average Feature**: A new variable was engineered by averaging the following:
  - `Age`
  - `Work/Study Hours`
  - `Financial Stress`

These variables showed a significant impact on depression predictions during the analysis.

---

## **Model Performance**
After testing various supervised learning models for this binary classification problem, **LightGBM** provided the best results. Below are the performance metrics:

| **Metric**         | **Score**          |
|---------------------|--------------------|
| **Accuracy**        | 0.9376            |
| **F1 Score**        | 0.8232            |
| **Precision**       | 0.8559            |
| **Recall**          | 0.7930            |
| **ROC-AUC Score**   | 0.8815            |

---

## **Highlights**
1. **Feature Importance**: Age, Work/Study Hours, and Financial Stress were the most significant predictors of depression.
2. **Optimal Model**: LightGBM was selected for its excellent balance of accuracy and speed.
3. **Robust Metrics**: The model achieved high accuracy and significant F1 and ROC-AUC scores, ensuring reliable predictions.

---

## **Conclusion**
This project demonstrates a complete pipeline for solving a mental health prediction problem using data preprocessing, feature engineering, and advanced machine learning techniques. The LightGBM model's performance highlights its suitability for binary classification tasks in such datasets. Further improvements can be made by incorporating more diverse data and fine-tuning the model further.

Feel free to explore the dataset and methodology to contribute to enhancing the solution!
