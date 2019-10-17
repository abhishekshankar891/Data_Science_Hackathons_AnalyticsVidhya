# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:18:12 2019

@author: U S PRASAD
"""

import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import sklearn
import seaborn as sns
from seaborn import countplot
from matplotlib.pyplot import figure, show

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

le1 = preprocessing.LabelEncoder()




data=pd.read_csv("data.csv")
test_df=pd.read_csv('final.csv')
#data['CapitalGain'].value_counts()

data.Relationship = data.Relationship.str.replace(' ', '')
data['Education']=data.Education.str.replace(' ', '')
data['MaritalStatus']=data.MaritalStatus.str.replace(' ', '')
data.WorkClass = data.WorkClass.str.replace(' ', '')
data.Gender=data.Gender.str.replace(' ', '')
data.NativeCountry =data.NativeCountry.str.replace(' ', '')
data['Occupation']=data['Occupation'].str.replace(' ', '')
data['Income']=data['Income'].str.replace(' ', '')

m1 = (data['Gender'].isnull()) & (data['Relationship'] == 'Wife')
m2=(data['Gender'].isnull()) & (data['Relationship'] == 'Husband')


data.loc[m1,'Gender'] = data.loc[m1,'Gender'].fillna('Female')


data.loc[m2,'Gender'] = data.loc[m2,'Gender'].fillna('Male')
#xx=data[((data['Relationship']=='Wife') | (data['Relationship']=='Husband')) &(data['Gender'].)isnull())]

dict1=dict({'Preschool':1,'1st-4th':2,'5th-6th':3,'7th-8th':4,'9th':5,'10th':6,'11th':7,'12th':8,'HS-grad':9,'Some-college':10,'Assoc-voc':11,'Assoc-acdm':12,'Bachelors':13,'Masters':14,'Prof-school':15,'Doctorate':16})


data['EducationNum']=data['Education'].map(dict1)


data = data.drop(columns=['Unnamed: 0'], axis=1)
data = data[(data != '?').all(axis=1)]


data["Income"]= data["Income"].replace("<=50K.", "<=50K")
data["Income"]= data["Income"].replace(">50K.", ">50K")


data.fillna(-999,inplace=True)

test_df.Relationship = test_df.Relationship.str.replace(' ', '')
test_df['Education']=test_df.Education.str.replace(' ', '')
test_df['MaritalStatus']=test_df.MaritalStatus.str.replace(' ', '')
test_df.WorkClass = test_df.WorkClass.str.replace(' ', '')
test_df.Gender=test_df.Gender.str.replace(' ', '')
test_df.NativeCountry =test_df.NativeCountry.str.replace(' ', '')
test_df['Occupation']=test_df['Occupation'].str.replace(' ', '')
test_df = test_df[(test_df != '?').all(axis=1)]
test_df.fillna(-999,inplace=True)




train_df=data

x = train_df.drop('Income',axis=1)
y = train_df.Income
y=le1.fit_transform(y)
cate_features_index = np.where(x.dtypes != float)[0]





xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=.75,random_state=1234)

model = CatBoostClassifier(iterations=1500, learning_rate=0.01, l2_leaf_reg=3.5, depth=8, rsm=0.98, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42)

#cv_data = cv(model.get_params(),Pool(x,y,cat_features=cate_features_index),fold_count=10)

model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))

test_df=test_df.iloc[:,1:]
test_cate_features_index = np.where(test_df.dtypes != float)[0]
#test_df=test_df.drop(columns=['Unnamed: 0','CapitalGain','CapitalLoss','Age','fnlwgt','EducationNum','HoursPerWeek'], axis=1)
test_data = Pool(data=test_df,
                 cat_features=test_cate_features_index)    
#pred = model.predict_proba(test_data)
pred = model.predict(test_data)
pred=le1.inverse_transform(pred.astype(int))
submission = pd.DataFrame({'Predicted_Income':pred})
submission.to_csv("Prediction1.csv",index=False)
pred1=model.predict_proba(xtest)[:,1]

###########ROC Curve Plot#########

roc_auc_score(ytest, pred1)

fpr, tpr, thresholds = roc_curve(ytest, pred1)
#  plot ROC_AUC Plot
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")

plt.savefig("ROCCurve.jpg")

#################

pred1=pred1.astype(int)
##############Confusion matrix##############
k = pd.DataFrame(sklearn.metrics.confusion_matrix(ytest,pred1))
accuracy=(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])
model.get_feature_importance(prettified=True)

feature_importance_df = pd.DataFrame(model.get_feature_importance(prettified=True), columns=['Feature Index', 'Importances'])
plt.figure(figsize=(12, 6));
sns.barplot(x="Importances", y="Feature Index", data=feature_importance_df);
plt.title('Features importance:');
plt.savefig("FeatureImportanc1e.jpg")


sklearn.metrics.roc_auc_score(ytest,pred1)

fpr, tpr, thresholds = sklearn.metrics.roc_curve(ytest,pred1, pos_label=0)
plt.plot(fpr,tpr)
plt.show() 

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)


###############PLots#######
xx=train_df[['NativeCountry','Income']]

#xx=xx[(xx.Education != -999)&(xx.NativeCountry !=-999)]

xx=xx[xx.NativeCountry != -999]

#fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(15,20))
grp=xx.groupby(['NativeCountry'])['Income'].count()
grp=grp.reset_index()
sns.countplot(xx['NativeCountry'],hue=xx['Income'])
sns.countplot(df['relationship'],hue=df['income'],ax=b)
sns.countplot(df['marital.status'],hue=df['income'],ax=c)
sns.countplot(df['race'],hue=df['income'],ax=d)
sns.countplot(df['sex'],hue=df['income'],ax=e)
sns.countplot(df['native.country'],hue=df['income'],ax=a)
#train_df['Occupation'].unique()
grp=xx.groupby(['NativeCountry'])['Income'].count()
grp=grp.reset_index()
grp.plot.bar()
grp.to_csv("CountryVsIncome.csv")

####Countplot
xx=train_df[['NativeCountry','Income']]
xx=xx[xx.NativeCountry != -999]
figure()
countplot(data=xx,y=xx.Income)
show()

xx=train_df[['Gender','Income']]
xx=xx[xx.Gender !=-999]
#figure(figsize=(12,6))
svm=countplot(data=xx,x=xx.Income, hue="Gender")
figure = svm.get_figure()    
figure.savefig('gender_income.png', dpi=400)
show()


xx=train_df[['WorkClass','Income']]

xx=xx[xx.WorkClass !=-999]

svm=countplot(data=xx,x=xx.Income, hue="WorkClass")
figure = svm.get_figure()    
figure.savefig('Work_income.png', dpi=400)
show()


xx=train_df[['HoursPerWeek','Income','NativeCountry']]

xx=xx[xx.NativeCountry !=-999]
#yy=xx[xx['WorkClass']=='Private']
svm=sns.catplot(x="NativeCountry", y="HoursPerWeek", hue="Income", kind="swarm", data=xx);
figure = svm.get_figure()    
figure.savefig('hours_income_cat.png', dpi=400)
show()
train_df.columns
