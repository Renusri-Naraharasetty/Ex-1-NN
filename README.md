<H3>ENTER YOUR NAME:RENUSRI NARAHARASHETTY</H3>
<H3>ENTER YOUR REGISTER NO.:212223240139</H3>
<H3>EX. NO.1</H3>
<H3>DATE:07/04/2025</H3>
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
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/sample_data/california_housing_train.csv')
print(df)
print("\n")

x=df.iloc[:,:-1].values
print(x)
print("\n")

y=df.iloc[:,-1].values
print(y)
print("\n")

print(df.isnull().sum())
print("\n")

df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
print("\n")

y=df.iloc[:,-1].values
print(y)
print("\n")

df.duplicated()
print(df['population'].describe())
print("\n")

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
print("\n")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print("\n")
print(x_test)
print(len(x_test))
```


## OUTPUT:
![image](https://github.com/user-attachments/assets/a11e19e6-8e4b-4834-ac92-baaff84e1ab8)

![image](https://github.com/user-attachments/assets/d0987bdc-8cbf-4a52-b443-3387c101d711)

![image](https://github.com/user-attachments/assets/4cc6091a-6555-47c0-9920-57dd7a47d3c8)

![image](https://github.com/user-attachments/assets/b47e5024-c17a-47e9-8074-0818e454a7dc)

![image](https://github.com/user-attachments/assets/1f1b68f8-cd94-4099-8335-079c7c39feaa)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


