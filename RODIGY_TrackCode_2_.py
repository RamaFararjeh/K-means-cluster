import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

df=pd.read_csv(r'C:\Users\admin\Development\Classification\Machine-Learning\Data-Analysis\K-means-cluster\Mall_Customers.csv')
print(df.head())
print(df.shape) # (200, 5)
print(df.describe())


# Check for missing values.

print(df.isna().sum()) # There is no missing values in this data farm.

# Histogram 
def histogram():
    for i in df:
        sns.histplot(df[i],bins=20,kde=True)
        plt.hist(df[i],bins=20,edgecolor='k',color='red')
        plt.xlabel(i)
        plt.ylabel('Frequancy')
        plt.title(f'Histogram for {i}')
        plt.show()

# histogram()

# Convert Categorical features to Numarical Using Label Encoder.
labelEncoder=LabelEncoder()
col_to_encode=['Gender']
df['Gender']=labelEncoder.fit_transform(df['Gender'])
print(df.head())

wcss=[] # Within-Cluster Sum of Squares (WCSS)
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(df)
    wcss.append(km.inertia_)
#The elbow curve
plt.figure(figsize=(8,6))
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

# Taking 5 Cluster
km1=KMeans(n_clusters=5)
# Fit
km1.fit(df)
# Predict
y=km1.predict(df)
# Add new col with cluster
df['Cluster']=y
print(df.head())

cust_0=df[df['Cluster']==0]
# print(cust_0)
print('Number of customer in 1st group =',len(cust_0))
print(f"Thay are {cust_0['CustomerID'].values}")
print('-'*50)
cust_1=df[df['Cluster']==1]
# print(cust_1)
print('Number of customer in 2nd group =',len(cust_1))
print(f"Thay are {cust_1['CustomerID'].values}")
print('-'*50)
cust_2=df[df['Cluster']==2]
# print(cust_2)
print('Number of customer in 3rd group =',len(cust_2))
print(f"Thay are {cust_2['CustomerID'].values}")
print('-'*50)
cust_3=df[df['Cluster']==3]
# print(cust_3)
print('Number of customer in 4th group =',len(cust_3))
print(f"Thay are {cust_3['CustomerID'].values}")
print('-'*50)
cust_4=df[df['Cluster']==4]
# print(cust_4)
print('Number of customer in 5th group =',len(cust_4))
print(f"Thay are {cust_4['CustomerID'].values}")

print('**************************************')

# NOTE : It creates a new DataFrame containing only the rows where the 'Cluster' column has a value of 1.
print(df[df['Cluster']==1])

# NOTE :  creates a Boolean Series indicating whether the condition is met for each row .
print(df['Cluster']==1)