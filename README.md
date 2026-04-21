<H3>ENTER YOUR NAME: JAWAHAR BABU S</H3>
<H3>ENTER YOUR REGISTER NO: 212224220041 </H3>
<H3>EX. NO.1</H3>
<H3>DATE: 21/04/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv("/content/Churn_Modelling.csv")
df

df.isnull().sum()

#check for duplication
df.duplicated()

print(df['CreditScore'].describe())

df.info()

df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df

Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

X = df1.iloc[:, :-1].values
print(X)

y = df1.iloc[:,-1].values
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```


## OUTPUT:
<img width="1270" height="389" alt="image" src="https://github.com/user-attachments/assets/18ceb200-c7f3-4344-8047-5fee191d1dd7" />
<img width="195" height="489" alt="image" src="https://github.com/user-attachments/assets/015b2a32-3653-4378-9506-b0a9eff43304" />
<img width="216" height="467" alt="image" src="https://github.com/user-attachments/assets/e206f614-e67e-4db3-9e22-4e1b73683546" />
<img width="1087" height="856" alt="image" src="https://github.com/user-attachments/assets/00ce57c8-35ef-49a6-a646-4cdec203c202" />
<img width="751" height="395" alt="image" src="https://github.com/user-attachments/assets/07e5b811-a134-4cb5-adc9-7b35ef15a8e5" />
<img width="639" height="232" alt="image" src="https://github.com/user-attachments/assets/f8c5912f-0eb0-4584-a98a-4ad1f60b822b" />
<img width="389" height="34" alt="image" src="https://github.com/user-attachments/assets/ad4a38a6-fc7b-425b-b200-b30e54511ad0" />
<img width="690" height="151" alt="image" src="https://github.com/user-attachments/assets/2cadf28a-837e-4877-a499-a057d1793425" />
<img width="730" height="149" alt="image" src="https://github.com/user-attachments/assets/6ad6517d-49a1-49e2-8a1d-1bc9f1b75d3e" />




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


