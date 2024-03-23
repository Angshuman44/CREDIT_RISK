## Credit Card Approval Prediction Project

This repository contains a project focused on predicting individuals' chances of credit card approval. Leveraging various machine learning techniques, the project aims to provide insights into the likelihood of an individual's credit card application being approved by financial institutions.

### Data Input Requirements

The project requires input data consisting of various demographic and financial attributes. These attributes are collected through a web form and include:

- **Gender**: The gender of the applicant.
- **Car Owner**: Whether the applicant owns a car.
- **Property Owner**: Whether the applicant owns any property or real estate.
- **Number of Children**: The number of dependents or family members supported by the applicant.
- **Annual Income**: The annual income of the applicant.
- **Type of Income**: The type of income earned by the applicant (e.g., Commercial associate, Pensioner, Working, State servant).
- **Education Level**: The highest level of education attained by the applicant.
- **Marital Status**: The marital status of the applicant.
- **Housing Type**: The type of housing the applicant resides in.
- **Age in Years**: The age of the applicant.
- **Years Employed**: The number of years the applicant has been employed.
- **Number of Family Members**: The total number of family members supported by the applicant.

By analyzing these features and their relationships with credit card approval decisions, the project aims to develop predictive models that can estimate the probability of an individual's credit card application being approved. Through this analysis, users can gain insights into the factors influencing credit card approval decisions and make informed decisions regarding their financial choices.

## Model Performance Summary

Here's a summary of the performance of various machine learning models on our dataset:

- **Support Vector Machine**: Accuracy - 77.60%
- **Logistic Regression**: Accuracy - 63.64%
- **XGBoost**: Accuracy - 90.26%
- **Random Forest**: Accuracy - 91.23%
- **Decision Tree**: Accuracy - 84.74%

Based on these results, the Random Forest classifier emerges as the top-performing model with an accuracy of 91.23%. This model outperforms others in accurately predicting outcomes on the dataset.
The prediction app was made using Flask, and the webapp has the interface as below:
![Screenshot from 2024-03-23 21-24-30](https://github.com/Angshuman44/CREDIT_RISK/assets/113175952/c559cc81-c873-4e17-b393-395db195d319)



Please follow the steps in order to use the model:
- a) Clone the repository
- b) run setup.py
- c) run app.py
- d) open localhost/predictdata
