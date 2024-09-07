import pandas as pd
import numpy as np
import matplotlib.pyplot as pit
import seaborn as sns
%matplotlib inline 
tit=pd.read_csv('full.csv')
tit

tit.info()

tit.describe().round(1)

tit.isna().sum()

sns.heatmap(tit.isna())

tit['Age']=tit['Age'].fillna(tit['Age'].mean())
sns.heatmap(tit.isna())

tit.drop('Cabin',axis=1,inplace=True)
sns.heatmap(tit.isna())

tit.head()

tit.info()

tit['Sex'].value_counts()

tit['Embarked'].value_counts()

sex=pd.get_dummies(tit['Sex'])
sex

emb=pd.get_dummies(tit['Embarked'])
emb

tit =pd.concat([tit,sex,emb],axis=1)
tit

print(tit.columns)


# للتحقق من الأعمدة المتاحة في DataFrame
print(tit.columns)
# حذف الأعمدة مع التعامل مع الأعمدة غير الموجودة باستخدام `errors='ignore'`
tit.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True, errors='ignore')



tit



tit.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked','Fare','WikiId','Name_wiki','Age_wiki','Boarded','Destination'], axis=1, inplace=True, errors='ignore')
tit


tit['Sex'] = tit.apply(lambda row: 'female' if row['female'] == 1 else 'male', axis=1)
# إنشاء المخطط باستخدام العمود الجديد
sns.countplot(x='Survived', data=tit, hue='Sex', palette='pastel')


tit['Age'].hist(bins=30)



tit






X=tit.drop('Survived',axis=1)
y=tit['Survived']
X

y




from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X , y , test_size=0.2 , random_state=42)
import pandas as pd
from sklearn.model_selection import train_test_split
# لنفترض أن لديك DataFrame يسمى tit
# تعريف X و y
X = tit.drop(columns=['Survived'])
y = tit['Survived']
# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test


X_train.shape

X_test.shape

 y_train.shape

 y_test.shape



from sklearn.linear_model import LogisticRegression 
model= LogisticRegression(max_iter=5000) #model building
model.fit(X_train,y_train) # model training


y_pre= model.predict(X_test) #model prediction (model testing)
y_pre


y_test.values




from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pre)



from sklearn.metrics import classification_report
print (classification_report(y_test,y_pre))



#الحمد الله 


