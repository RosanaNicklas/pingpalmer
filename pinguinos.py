import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('penguins_lter.csv')
print(df)

print(df.head)
#INFO

print(df.info())
print(df.describe())

#BORRO
df.drop(columns=['Comments','Delta 15 N (o/oo)','Delta 13 C (o/oo)','Date Egg','Individual ID','Stage','Region'],axis=1,inplace=True)

#SPECIES A NUMERAL
for i in range(len(df['Species'])):
    if df['Species'][i] =='Adelie Penguin (Pygoscelis adeliae)':
        df['Species'][i]=0
    elif df['Species'][i] =='Chinstrap penguin (Pygoscelis antarctica)':
        df['Species'][i]=1
    else:
        df['Species'][i]=2
print(df)

#ISLAND A NUMERAL
for i in range(len(df['Island'])):
    if df['Island'][i] =='Torgersen':
        df['Island'][i]=0
    elif df['Island'][i] =='Biscoe':
        df['Island'][i]=1
    else:
        df['Island'][i]=2
print(df)

#CLUTCH COMPLETION A NUMERAL
df['Clutch Completion'].unique()
for i in range(len(df['Clutch Completion'])):
    if df['Clutch Completion'][i] =='Yes':
        df['Clutch Completion'][i]=1
    
    else:
        df['Clutch Completion'][i]=0
print(df)

plt.figure(figsize=(6, 6))

cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
sns.heatmap(df.isnull(), cmap=cmap)

#VALORES NULOS
df= df.dropna().reset_index()
print(df.info())


plt.figure(figsize=(6, 6))

cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse=True)
sns.heatmap(df.isnull(), cmap=cmap)

#SEXO A NUMERAL
df['Sex'].astype(str)
for j,i  in enumerate(df['Sex']):
       
    
    if i == 'FEMALE':
        df['Sex'][j]=0
    elif i=='MALE':
        df['Sex'][j]=1
    
    else:
        df['Sex'][j]=1


df1 = df.iloc[:,3:]
print(df1.head())

# CAMBIAR FORMATO DE DATOS
cat_val = ['Species' , 'Island','Clutch Completion' , 'Sex']
from sklearn.preprocessing import LabelEncoder
for i in cat_val:
    le = LabelEncoder()
    df1[i] = le.fit_transform(df1[i])

print(df1.dtypes)   

#SEPARAR DATOS Y OBJETIVO
y = df1['Sex']
df1.drop(columns = ['Sex'],axis=1,inplace=True)
x = df1
print('Input Data -',x.head())
print('Target Data-',y.head())

#FORMA
a= x.values
print(a.shape)

#CORRELACION
corr = np.corrcoef(a.T)
print(corr.shape)

plt.figure(figsize = (9,9))
sns.heatmap(corr,cmap='Blues')
plt.title("Correlation Matrix")
plt.show()
print('This figure shows that  5 & 6 are highly correlated')

#ENTRENANDO
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.30,stratify=y,random_state=50)


#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
acc = knn.score(X_test,y_test)
print('The accuracy of KNN on the original dataset : {}'.format(acc))

#Comparando
# Comparing the result of KNN classifier after LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda2 = LinearDiscriminantAnalysis(n_components = 1) # down scaling the dimension to 1D from original data
lda2.fit(X_train,y_train)
x_train_mod = lda2.transform(X_train)
x_test_mod  = lda2.transform(X_test)

#EXITO?
knn.fit(x_train_mod,y_train)
acc = knn.score(x_test_mod,y_test)
print('The accuracy of KNN after LDA on the original dataset : {}'.format(acc))
