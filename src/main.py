# Import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/train.csv')

# Print column names
print(data.columns)

# Drop unnecessary columns
data = data.drop(['PassengerId', 'Survived', 'Name',
                 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Convert 'Sex' column to numerical data
data['Sex'] = np.where(data['Sex'] == 'male', 1, 0)

# Replace missing values in 'Age' column with mean age
data['Age'] = data['Age'].fillna(data['Age'].mean())

# Standardize data
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# Determine optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_std)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), wcss)
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Fit K-Means algorithm to data
kmeans = KMeans(n_clusters=3, init='k-means++',
                max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(data_std)

# Visualize clusters
plt.scatter(data_std[:, 0], data_std[:, 1], c=pred_y)
plt.title('Clusters')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()


# scatter plot
import seaborn as sns  # nopep8

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Replace 'male' and 'female' with 1 and 0 in the 'sex' column
titanic['sex'] = pd.factorize(titanic['sex'])[0]

# Create a scatter plot of age vs. fare, with color-coded groups based on passenger class
sns.scatterplot(x='age', y='fare', hue='class', data=titanic)
plt.title('Titanic Dataset')
plt.show()
