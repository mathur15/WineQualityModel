#This is just an intro script to familiarize myself with the core functions
#releated to machine learning in python
#The goal is to assess the quality of wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#family of model - RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib

#import and load data
url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep = ";")

#fix data to fit the need (standardize data)
#print(data.head(5))
#print(data.shape)
#print(data.describe())

#split data into training and test sets using train_test_split
y = data.quality
X = data.drop('quality', axis = 1)

#stratify sample by the target variable y 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 123,
                                                    stratify = y)

#standardize data using the Tranformer API to use means and sd for future 
#datasets
transformation_values = preprocessing.StandardScaler().fit(X_train)

#apply tranformation on X_train - expect to centered around 0 ( mean) and 1(sd)

X_train_scaled = transformation_values.transform(X_train)
#print(X_train_scaled.mean(axis=0))
#print(X_train_scaled.std(axis=0))

#apply transformation to X_test- mean and std wont be exactly 0 and 1 as this 
#transformation is based on the X_train

X_test_scaled = transformation_values.transform(X_test)
#print(X_test_scaled.mean(axis=0))
#print(X_test_scaled.std(axis=0))

#instead create a pipeline and apply the RandomForest
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators = 100))

#gives us the hyperparameter to choose from
#print( pipeline.get_params())

#hyperparameters to tune
hyperparameters = {'randomforestregressor__max_features':['auto','sqrt','log2'],
                   'randomforestregressor__max_depth' :[None,5,3,1]}

#tune model using cross validation pipeline using the training set
  #-to evaluate a method in this case hyperparameters and measure effectiveness
  #-make a cross validation pipeline

cross_pipeline = GridSearchCV(pipeline, hyperparameters,cv=10)
#print(cross_pipeline)
#divided data 10 folds and preprocess and train on k-1 folds k times where k=10

cross_pipeline.fit(X_train,Y_train)
#print(cross_pipeline.best_params_)

#refit the entire training set
#print(cross_pipeline.refit)

#Evaluate cross_pipeline on the test data

y_predict = cross_pipeline.predict(X_test)

#evaluate performance on the test data
print(r2_score(Y_test, y_predict))
print(mean_squared_error(Y_test, y_predict))

#save the model
joblib.dump(cross_pipeline,'Model.pkl')

#Ways to improve the model
#Try other regression model families (e.g. regularized regression, boosted trees, etc.).
#Collect more data if it's cheap to do so.
#Engineer smarter features after spending more time on exploratory analysis.
#Speak to a domain expert to get more context













