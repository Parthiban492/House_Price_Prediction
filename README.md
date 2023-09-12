# House_Price_Prediction
The purpose of the project is to develop a Machine Learning Model that can accurately estimate the prices of house based on various features.The model's aim is generalize well to unseen data and aid potential buyers,sellers and real estate proffessionals in making informed decision about house pricing.

## <img src=https://user-images.githubusercontent.com/106439762/181935629-b3c47bd3-77fb-4431-a11c-ff8ba0942b63.gif height=45 width=45> Data Dictionary:

| Files | Description |
|-------| ------------|
| Dataset | This folder houses a comprehensive collection of features and details of the Houses. |
| Code | This folder encompasses the code and model for the House_Price_Prediction project. |
| Presentation | This folder encompasses the PowerPoint presentation of the project. |
| Webpage | This folder encompasses the code used to develop the webpage and link of webpage of the project. |
| README.md | This is the Readme file of the project. |

<be>

## About this Dataset: 
The dataset contains housing information with columns such as date, price, bedrooms, bathrooms, and more. It includes details like the number of bedrooms and bathrooms, square footage of living space, lot size, and the number of floors. Additionally, it includes features like waterfront status, condition, and grade, along with information about the year built and renovated. The dataset also provides geographical data with latitude and longitude coordinates, as well as neighborhood-related features such as sqft_living15 and sqft_lot15, which represent the living area and lot size of nearby properties.
<be>


## <img src=https://user-images.githubusercontent.com/106439762/181937125-2a4b22a3-f8a9-4226-bbd3-df972f9dbbc4.gif height=45 width=45> Objective:
The objective of this project is to develop a machine learning model that can predict house prices based on various features such as the number of bedrooms, bathrooms, square footage, location, and other relevant factors, ultimately assisting homebuyers and sellers in estimating property values accurately.
<be>

## <img src=https://user-images.githubusercontent.com/106439762/178428775-03d67679-9aa4-4b08-91e9-6eb6ed8faf66.gif height=45 width=45> Libraries Used:
`Pandas (import pandas as pd):`` Pandas is a powerful data manipulation and analysis library. It was used for data loading, data cleaning, data exploration, and data preprocessing tasks. Pandas provides data structures like DataFrames, which make it convenient to work with tabular data.

`Matplotlib (import matplotlib.pyplot as plt):` Matplotlib is a popular data visualization library in Python. It was utilized to create various plots and charts to visualize the data distribution, relationships, and model performance.

`Seaborn (import seaborn as sns):` Seaborn is built on top of Matplotlib and provides a higher-level interface for creating informative and attractive statistical graphics. It enhances the aesthetics of plots and simplifies complex visualizations.

`Scipy (from scipy.stats import chi2_contingency):` Scipy is a library for scientific computing in Python. In this project, it was used for conducting the chi-squared test of independence using the chi2_contingency function to analyze categorical variable relationships.

`Scikit-learn (from sklearn):` Scikit-learn is a versatile machine learning library that provides tools for data preprocessing, model selection, training, and evaluation. Specific modules and classes used include:

`train_test_split:` Used for splitting the dataset into training and testing sets.
`StandardScaler:` Employed for feature scaling to ensure consistent feature ranges.
`LinearRegression:` Used to build a linear regression model.
`DecisionTreeRegressor:` Utilized for decision tree-based regression modeling.
`RandomForestRegressor:` Employed for random forest regression modeling.
`SVR (Support Vector Regression):` Used to implement support vector regression.
`mean_squared_error and r2_score:` Metrics for evaluating model performance.
`Statsmodels (from statsmodels.stats.outliers_influence import variance_inflation_factor):` Statsmodels is a library for estimating and interpreting statistical models. In this project, it was used to calculate the variance inflation factor (VIF) to assess multicollinearity among features.

These libraries collectively provide a comprehensive toolkit for data analysis, preprocessing, modeling, and evaluation in your "House Price Prediction" project.
<be>

## <img src=https://user-images.githubusercontent.com/106439762/181937125-2a4b22a3-f8a9-4226-bbd3-df972f9dbbc4.gif height=45 width=45> Data Hangeling/Preprocessing:
Before training the machine learning model, the dataset underwent thorough data cleaning and preprocessing to ensure its quality and suitability for analysis. The following steps were carried out:

`Handling Missing Values:` Any missing values in the dataset, if present, were addressed. This might involve filling missing values with appropriate values or removing rows/columns with excessive missing data.

`Data Type Conversion:` Ensuring that data types of columns are appropriate for analysis. For example, converting date columns to datetime objects, if needed.

`Outlier Detection and Handling:` Identifying and dealing with outliers in numerical features using statistical methods or domain knowledge.

`Feature Engineering:` Creating new features or transforming existing ones to extract meaningful information. For instance, combining 'sqft_above' and 'sqft_basement' to create a 'total_sqft' feature.

`Categorical Variable Encoding:` Converting categorical variables into a numerical format using techniques like one-hot encoding or label encoding.

`Normalization/Scaling:` Scaling numerical features to ensure they have similar ranges, which can be crucial for certain machine learning algorithms.

`Feature Selection:` Selecting the most relevant features for model training to improve efficiency and reduce noise in the dataset.

`Data Splitting:` Dividing the dataset into training and testing sets to assess model performance accurately.

By performing these data cleaning and preprocessing steps, we ensured that the dataset was in a suitable and reliable state for training machine learning models, ultimately improving the accuracy and effectiveness of our house price prediction model.
<be>

## <img src=https://user-images.githubusercontent.com/106439762/178428775-03d67679-9aa4-4b08-91e9-6eb6ed8faf66.gif height=45 width=45> ML Models:
### Code Used:
```ruby
# Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the Variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and Evaluate Models (after removing high VIF features)
models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), SVR()]
model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR']

for i, model in enumerate(models):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{model_names[i]} - MSE: {mse}, R-squared: {r2}, MAE: {mae}, RMSE: {rmse}")
```
Based on the evaluation metrics, it is evident that the Random Forest Regression model outperformed the other models in terms of both R-squared (closer to 1 is better) and lower RMSE (indicating better predictive accuracy). Consequently, the Random Forest model was selected and developed further due to its superior performance.

This selection was made because the Random Forest model demonstrated the highest predictive power, capturing more complex relationships within the data and providing more accurate house price predictions.
<be>

## <img src=https://user-images.githubusercontent.com/106439762/181937125-2a4b22a3-f8a9-4226-bbd3-df972f9dbbc4.gif height=45 width=45> Website build:

<img src=https://github.com/Parthiban492/House_Price_Prediction/blob/430fac3cc251894976203ef1929ceb265d41f06d/Presentation/Screen%20shot%20of%20the%20website%20(1).jpeg height=800 width=1200>

**Note:** The screenshot above provides a visual representation of our house price prediction website. Users can input property details and receive predictions for house prices using our machine-learning model.

**Price-Prediction based on required feature**
<img src=https://github.com/Parthiban492/House_Price_Prediction/blob/main/Presentation/Output.png height=800 width=1200>
<be>

## Conclusion:

In this project, we successfully developed a robust house price prediction model that leverages machine learning techniques to provide accurate estimates for property values. We systematically cleaned and preprocessed the data, explored various machine learning algorithms, and ultimately selected the Random Forest Regression model for its exceptional performance. The creation of a user-friendly web interface further enhances the project's usability and practicality. We believe this project will be a valuable resource for individuals navigating the real estate market, and we are excited to see its impact.
<be>

## Challenges Faced:

Throughout the course of this project, we encountered several challenges that tested our problem-solving skills and determination. Some of the key challenges we faced included:

`Data Quality:` The dataset presented issues with missing values and outliers, requiring careful handling and cleaning.

`Feature Engineering:` Extracting meaningful features and deciding which variables to include in the model was a complex task.

`Model Selection:` Identifying the most suitable machine learning algorithm among various options demanded thorough experimentation and evaluation.

`Web Development:` Designing and implementing an intuitive web interface that seamlessly integrates with the machine learning model was a multifaceted task.

`Performance Tuning:` Achieving the desired level of prediction accuracy while optimizing model performance was an ongoing challenge.

<be>

## Future Scope:

While we have achieved significant milestones in this project, there are several avenues for future exploration and enhancement:

`Feature Engineering:` Continuously refining and expanding feature engineering techniques can potentially improve model accuracy.

`Hyperparameter Tuning:` Conducting a more exhaustive search for optimal hyperparameters in the selected model could lead to further performance improvements.

Additional Data Sources:` Incorporating additional datasets, such as economic indicators or neighborhood-specific information, could enhance the model's predictive capabilities.

`Real-Time Data:` Developing mechanisms to incorporate real-time data updates for more up-to-date predictions.

`Deployment:` Expanding the deployment of the web interface, making it accessible to a broader audience, and incorporating user feedback for continuous improvement.

`Interpretability:` Implementing model interpretability techniques to provide users with insights into how predictions are generated.

This project serves as a solid foundation for future research and development, and we look forward to seeing how it evolves to better serve the needs of users in the dynamic real estate market.

## Reference:
https://house-price-predictions.netlify.app/
