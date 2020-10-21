import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# Cleaning of Data

# Check unavailable data
# print("train", train.isna().head().sum())
# print("test", test.isna().head().sum())

# fillna() fill missing values, with means works only for numerical value
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)
# print("train", train.isna().head().sum())
# print("test", test.isna().head().sum())
# print(train['Ticket'].head(), train['Cabin'].head())


# Survival count respect to Pclass
print(
    train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                         ascending=False))
# Survival count respect to Sex
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# Age vs Survived
def age_vs_sur():
    g = sns.FacetGrid(train, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    plt.show()


# age_vs_sur()

# Pcalss vs Survived
def class_vs_sur():
    g = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
    g.map(plt.hist, 'Age', alpha=.5, bins=20)
    plt.show()


# class_vs_sur()

# print(train.info())

# Remove non numerical features
train = train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Label Encoding trasform into numerical
label_encoder = LabelEncoder()
label_encoder.fit(train['Sex'])
label_encoder.fit(test['Sex'])

train['Sex'] = label_encoder.transform(train['Sex'])
test['Sex'] = label_encoder.transform(test['Sex'])

print(train.head(6))

# K-means
X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])
print(X)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

colors = ['#4eacc5', '#FF9C34', '#4E9A06']
kmeans_labels = kmeans.labels_
kmeans_cluster_center = kmeans.cluster_centers_
kmeans_labels_unique = np.unique(kmeans_labels)


def plot_k_means():
    plt.figure()
    for k, col in zip(range(3), colors):
        my_members = kmeans_labels == k
        cluster_center = kmeans_cluster_center[k]
        plt.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    plt.title('K-means')
    plt.grid(True)
    plt.show()


# plot_k_means()

# Evaluate
correct = 0
for i in range(len(X)):
    predict_me = np.array((X[i].astype(float)))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct / len(X))
