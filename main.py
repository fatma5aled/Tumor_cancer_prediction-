import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import streamlit as st
def set_background():
    st.markdown(
        """
        <style>
          .stApp {
             background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWSq4QSLT7shE3LBB5n_8Yyni9lpKIhXAqtA&usqp=CAU");
             background-attachment: fixed;
             background-size: cover;
             background-position:center;
             background-repeat: no-repeat;
             padding-top: 100px;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()
data=pd.read_csv('Tumor Cancer Prediction_Data.csv')
print(data.head())
x=data.iloc[:,:-1]
diagnosis=data.iloc[:,-1]
print(diagnosis)
label=LabelEncoder()
diagnosis=label.fit_transform(diagnosis)
print(diagnosis)
columns_with_nulls = data.columns[data.isnull().any()].tolist()
print(columns_with_nulls)
#don't have any nulls

data.drop_duplicates(inplace=True)
correlation_matrix = x.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
least_corr_columns = correlation_matrix.abs().sum().nsmallest(25).index
print("Columns with the least correlation:")
print(least_corr_columns)
Columns_with_least_correlation=['Index', 'texture_se', 'symmetry_se', 'smoothness_se', 'texture_mean',
       'texture_worst', 'symmetry_worst', 'fractal_dimension_se',
       'smoothness_worst', 'fractal_dimension_worst', 'symmetry_mean',
       'fractal_dimension_mean', 'smoothness_mean', 'concavity_se',
       'compactness_se','area_se', 'concave_points_se', 'radius_se',
       'perimeter_se', 'area_mean', 'radius_mean', 'compactness_worst',
       'area_worst', 'perimeter_mean', 'radius_worst']
for column in Columns_with_least_correlation:
    x=x.drop(column, axis=1)
print(x.shape)
print(x.columns)
X_train, X_test, y_train, y_test = train_test_split(x, diagnosis, test_size=0.2, random_state=0)
scaler=StandardScaler(copy=True,with_mean=True,with_std=True)
X_train=scaler.fit_transform(X_train)
poly=PolynomialFeatures(degree=2)
X_train=poly.fit_transform(X_train)
LogisticRegression_model = LogisticRegression()
param_log = {
    'C': [0.1, 1, 10],
     'solver': [ 'sag']
}
grid_search_logistic_reg = GridSearchCV(estimator=LogisticRegression_model, param_grid=param_log, cv=5)
grid_search_logistic_reg.fit(X_train, y_train)
best_params_logistic_reg = grid_search_logistic_reg.best_params_
print("Best parameters of logistic regession",best_params_logistic_reg)
best_model_logistic_reg = grid_search_logistic_reg.best_estimator_
joblib.dump(best_model_logistic_reg,"logistic_regression.pkl")
###################################################################3
DecisionTree_model = DecisionTreeClassifier()
param_DT = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}
grid_search_decisiontree = GridSearchCV(estimator=DecisionTree_model, param_grid=param_DT, cv=5)
grid_search_decisiontree.fit(X_train, y_train)
print("best_params_decisiontree",grid_search_decisiontree.best_params_)
best_model_decisiontree= grid_search_decisiontree.best_estimator_
joblib.dump(best_model_decisiontree,"decisiontree.pkl")
svc_model = SVC()
param_svc = {
    'C': [.001,0.1, 1, 10],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(estimator=svc_model, param_grid=param_svc, cv=5)
grid_search_svc.fit(X_train, y_train)
print("best_params_svc",grid_search_svc.best_params_)
best_model_svc= grid_search_svc.best_estimator_
joblib.dump(best_model_svc,"svc.pkl")
#######################################################################################################
X_test=scaler.transform(X_test)
joblib.dump(scaler,"scaler.pkl")
X_test=poly.transform(X_test)
joblib.dump(poly,"polynomial.pkl")
logistic=joblib.load("logistic_regression.pkl")
y_pred_logistic_reg = logistic.predict(X_test)
print("logistic regression Score:",logistic.score(X_test, y_test))
print('MSE of a logistic regression:',mean_squared_error(y_test,y_pred_logistic_reg))
print("logistic regression_prediction",y_pred_logistic_reg)
DT=joblib.load("decisiontree.pkl")
y_pred_decisiontree= DT.predict(X_test)
print("Decision_tree score",DT.score(X_test,y_test))
print("MSE of a Decision tree",mean_squared_error(y_test,y_pred_decisiontree))
print("Decision_tree_prediction",y_pred_decisiontree)
svc=joblib.load("svc.pkl")
y_pred_svc= svc.predict(X_test)
print("svc_prediction",y_pred_svc)
print("svc score",svc.score(X_test,y_test))
print("MSE of a svc",mean_squared_error(y_test,y_pred_svc))
scores = [logistic.score(X_test, y_test),DT.score(X_test,y_test),DT.score(X_test,y_test)]
labels = ['logistic_Regression Score ', 'decision_tree Score ', 'SVC Score ']
plt.bar(labels, scores)
plt.xlabel('Scores')
plt.ylabel('Values')
plt.title('Scores Visualization')
plt.show()
mses=[mean_squared_error(y_test,y_pred_logistic_reg),mean_squared_error(y_test,y_pred_decisiontree),mean_squared_error(y_test,y_pred_svc)]
labels2=["logistic_Regression  MSE","decision_tree MSE","SVC MSE"]
plt.pie(mses, labels=labels2, autopct='%1.1f%%')
plt.title('mse Visualization')
plt.show()
st.title("Predictions:")
compactness_mean=st.number_input("write your compactness_mean ")
concavity_mean=st.number_input("write your concavity_mean")
concave_points_mean=st.number_input("write your concave_points_mean")
concavity_worst=st.number_input("write your concavity_worst")
perimeter_worst=st.number_input("write your perimeter_worst")
concave_points_worst=st.number_input("write your concave_points_worst")
if st.button("Predict"):
 new_data=list([compactness_mean,concavity_mean,concave_points_mean,concavity_worst,perimeter_worst,concave_points_worst])
 new_data = [new_data]
 saved_scaler=joblib.load("scaler.pkl")
 saved_polynomial_feature=joblib.load("polynomial.pkl")
 new_data=saved_scaler.transform(new_data)
 new_data=saved_polynomial_feature.transform(new_data)
 prediction=[]
 firstmodel_prediction=logistic.predict(new_data)
 prediction.append(firstmodel_prediction)
 secmodel_prediction=DT.predict(new_data)
 prediction.append(secmodel_prediction)
 thirdmodel_prediction=svc.predict(new_data)
 prediction.append(thirdmodel_prediction)
 final_prediction=sum(prediction)  
 if final_prediction < 2:
     final_prediction="Benignant"
 else:
    final_prediction="Malignant"
 st.write("Logistic Regression:", prediction[0])    
 st.write("Decision Tree:", prediction[1])   
 st.write("SVM:",prediction[2])
 st.title("Final prediction")      
 st.write(final_prediction)

