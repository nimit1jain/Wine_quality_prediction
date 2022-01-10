from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
wine_data=pd.read_csv("N:\Machine learning\Algorithms\winequality-red.csv")
# print(wine_data.head())

                                    #--------check for null values--------

# print(wine_data.isnull().any())

# we found that no null values are present so we can move forward

                                    #--------check for outliers----------


for i in wine_data.columns:
    q75, q25 = np.percentile(wine_data[i], [75 ,25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
   
    wine_data=wine_data[(wine_data[i]<max_val)&(wine_data[i]>min_val)]


                                    #-----feature selection--------
# cmap=sns.diverging_palette(500,10, as_cmap=True)
# sns.heatmap(wine_data.corr(),annot=True)
# plt.show()

wine_data=wine_data.drop(['fixed acidity','residual sugar','free sulfur dioxide','pH'],axis=1)


#-----Now this problem can be converted to regression or classification problem-----

temp_data=wine_data
                                  #-----for classification problem we have to make classes (good=1,bad=0)-----

temp_data['Quality_mod']=[1 if x>=5 else 0 for x in temp_data['quality']]
# print(temp_data['Quality_mod'].value_counts())
features=temp_data.drop(['quality','Quality_mod'],axis=1)
target=temp_data['Quality_mod']

                                     #------for regression model just seperate target and features----
# target=temp_data['quality']
# features=temp_data.drop('quality',axis=1)


                                     #---------data scaling and splitting--------

scaler=MinMaxScaler()
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.3,random_state=0)
# features=scaler.fit_transform(features)
# target=scaler.fit_transform(target)
# target=pd.DataFrame(target)
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
x_test=pd.DataFrame(x_test)
x_train=pd.DataFrame(x_train)

def MAPE(y_actual,y_predicted):
    mape = np.mean(np.abs((y_actual - y_predicted)/y_actual))*100
    return mape
                                    #------Regression models------
                                
                                #-----KNN-------


# model=neighbors.KNeighborsRegressor(n_neighbors=32)
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)

# mse = mean_squared_error(y_test, y_pred)
# rmse=np.sqrt(mean_squared_error(y_test,y_pred))
# r2score = r2_score(y_test, y_pred)
# print('Model mse: ',mse)        # 0.3419442261464497
# print('Model rmse: ',rmse)      # 0.5847599731055895
# print('Model r2_score: ',r2score)    # 0.35254116660244295
# print('Model MAPE: ',MAPE(y_test,y_pred))    # 8.634474499859115
               
                                #---------Linear Regression-------

# model=LinearRegression()
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)

# mse = mean_squared_error(y_test, y_pred)
# rmse=np.sqrt(mean_squared_error(y_test,y_pred))
# r2score = r2_score(y_test, y_pred)
# print('Model mse: ',mse)        # 0.3295210938505716
# print('Model rmse: ',rmse)      # 0.5740392790137027
# print('Model r2_score: ',r2score)    # 0.37606391133212835
                                  
                                  #---------Classification models-----------


                                  #-----Logistic regression-----------
# model=LogisticRegression()
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)


# confusion_mat=confusion_matrix(y_test,y_pred,labels=None)
# print("confusion_mat = ",confusion_mat)
# print("Accuracy Score:",accuracy_score(y_test,y_pred))           # 0.7485207100591716
# print("precision score = ",precision_score(y_test, y_pred))       # 0.7988165680473372
# print("recall score = ",recall_score(y_test, y_pred))            # 0.7258064516129032
# print("F1 score = ",f1_score(y_test, y_pred))                    # 0.7605633802816901

                                      #----SVC with kernel='linear'-------
# model=SVC(kernel='linear')
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)


# confusion_mat=confusion_matrix(y_test,y_pred,labels=None)
# print("confusion_mat = ",confusion_mat)
# print("Accuracy Score:",accuracy_score(y_test,y_pred))              # 0.7366863905325444
# print("precision score = ",precision_score(y_test, y_pred))         # 0.8050314465408805
# print("recall score = ",recall_score(y_test, y_pred))               # 0.6881720430107527
# print("F1 score = ",f1_score(y_test, y_pred))                       # 0.7420289855072464


                                      #----SVC with kernel='rbf'-------
# model=SVC(C=2.2,kernel='rbf')
# model.fit(x_train,y_train)
# y_pred=model.predict(x_test)


# confusion_mat=confusion_matrix(y_test,y_pred,labels=None)
# print("confusion_mat = ",confusion_mat)
# print("Accuracy Score:",accuracy_score(y_test,y_pred))              # 0.757396449704142
# print("precision score = ",precision_score(y_test, y_pred))         # 0.8291139240506329
# print("recall score = ",recall_score(y_test, y_pred))               # 0.7043010752688172
# print("F1 score = ",f1_score(y_test, y_pred))                       # 0.7616279069767442


                                       #----KNN-------
model=neighbors.KNeighborsClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


confusion_mat=confusion_matrix(y_test,y_pred,labels=None)
print("confusion_mat = ",confusion_mat)
print("Accuracy Score:",accuracy_score(y_test,y_pred))              # 0.7307692307692307
print("precision score = ",precision_score(y_test, y_pred))         # 0.7653631284916201
print("recall score = ",recall_score(y_test, y_pred))               # 0.7365591397849462
print("F1 score = ",f1_score(y_test, y_pred))                       # 0.7506849315068495
