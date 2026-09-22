# K-Means Clustering – Customer Segmentation

## 📌 Project Overview

This project demonstrates **K-Means Clustering**, an **unsupervised machine learning algorithm**, using the **Mall Customers dataset**.

The objective is to group customers into different segments based on:

* **Annual Income (k$)**
* **Spending Score (1–100)**

The project uses the **Elbow Method** to determine a suitable number of clusters and then applies K-Means clustering to identify customer groups.

---

## 🎯 Project Objective

The main objectives are:

1. Understand K-Means Clustering.
2. Apply an unsupervised machine learning algorithm.
3. Identify customer groups with similar characteristics.
4. Use the **Elbow Method** to select an appropriate value of K.
5. Visualize customer clusters.
6. Understand cluster centroids.
7. Explore how changing K affects customer segmentation.

---

## 📂 Dataset

Dataset:

```text
Mall_Customers.csv
```

Typical columns:

| Column                 | Description                |
| ---------------------- | -------------------------- |
| CustomerID             | Unique customer identifier |
| Gender                 | Customer gender            |
| Age                    | Customer age               |
| Annual Income (k$)     | Annual income in thousands |
| Spending Score (1-100) | Customer spending score    |

For this project, we use only:

```text
Annual Income (k$)
Spending Score (1-100)
```

Python selection:

```python
X = dataset.iloc[:, [3, 4]].values
```

---

# 🤖 What is K-Means Clustering?

K-Means is an **unsupervised machine learning algorithm** used to divide data into **K groups called clusters**.

Each cluster contains data points that are relatively similar to each other.

For example:

```text
Customer Data
      |
      ↓
   K-Means
      |
      ↓
┌─────┬─────┬─────┬─────┬─────┐
│ C1  │ C2  │ C3  │ C4  │ C5  │
└─────┴─────┴─────┴─────┴─────┘
```

In this project, the customers are divided into groups based on income and spending behavior.

---

# 🔍 Supervised vs Unsupervised Learning

### Supervised Learning

The dataset contains a target/output variable.

Examples:

* Linear Regression
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Example:

```text
Input → Model → Known Target
```

### Unsupervised Learning

There is no predefined target variable.

The algorithm tries to discover patterns or groups within the data.

Examples:

* K-Means
* Hierarchical Clustering
* DBSCAN
* PCA

Example:

```text
Data → Algorithm → Hidden Groups/Patterns
```

K-Means belongs to **unsupervised learning**.

---

# 🧠 Important K-Means Concepts

## 1. K

`K` represents the number of clusters.

For example:

```python
KMeans(n_clusters=5)
```

means:

```text
Create 5 clusters
```

---

## 2. Centroid

A centroid represents the approximate center of a cluster.

Example:

```text
       ● ● ●
      ●  X  ●
       ● ●

       X = Centroid
```

K-Means continuously updates the centroids until the clustering stabilizes.

---

## 3. Distance

K-Means commonly uses **Euclidean distance**.

Formula:

```text
d = √((x₂-x₁)² + (y₂-y₁)²)
```

The algorithm assigns each data point to the nearest centroid.

---

# ⚙️ How K-Means Works

K-Means follows these basic steps:

### Step 1 – Select K

Choose the number of clusters.

```text
K = 5
```

### Step 2 – Initialize Centroids

Initial centroid positions are selected.

### Step 3 – Assign Data Points

Each customer is assigned to the nearest centroid.

### Step 4 – Recalculate Centroids

The center of each cluster is recalculated.

### Step 5 – Repeat

The assignment and centroid calculation are repeated.

### Step 6 – Convergence

The process stops when the centroids no longer change significantly or the maximum number of iterations is reached.

---

# 📉 What is WCSS?

WCSS means:

**Within-Cluster Sum of Squares**

It measures how close the data points are to their cluster centroids.

Conceptually:

```text
WCSS = Sum of squared distances
       between each point and
       its cluster centroid
```

Lower WCSS generally means that the points are more compact within their clusters.

However:

> WCSS will generally decrease as K increases.

Therefore, we don't simply select the K with the lowest WCSS.

---

# 📐 Elbow Method

The **Elbow Method** helps determine a suitable value of K.

We calculate WCSS for multiple K values:

```text
K = 1
K = 2
K = 3
...
K = 10
```

Then we plot:

```text
Number of Clusters vs WCSS
```

Example:

```text
WCSS
 |
 |\
 | \
 |  \
 |   \
 |    \__
 |       \__
 |          \___
 +--------------------> K
   1  2  3  4  5  6 ...
```

The point where the curve begins to flatten is called the **elbow**.

For this Mall Customers example, **K = 5 is commonly used because the elbow appears around this region**.

---

# 💻 Project Implementation

## Step 1 – Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
```

### Explanation

```python
import numpy as np
```

Used for numerical operations.

```python
import pandas as pd
```

Used for loading and manipulating the dataset.

```python
import matplotlib.pyplot as plt
```

Used for visualization.

```python
from sklearn.cluster import KMeans
```

Imports the K-Means algorithm from Scikit-Learn.

---

# Step 2 – Load Dataset

```python
dataset = pd.read_csv(
    r"D:\work\olama resm\ML work\K-Means Clustering\Mall_Customers.csv"
)
```

The CSV file is loaded into a Pandas DataFrame.

For GitHub, it is better to use a relative path:

```python
dataset = pd.read_csv("Mall_Customers.csv")
```

This makes the project easier for other users to run.

---

# Step 3 – Select Features

```python
X = dataset.iloc[:, [3, 4]].values
```

We select:

```text
Column 3 → Annual Income
Column 4 → Spending Score
```

So:

```text
X =
[
  [Income, Spending Score],
  [Income, Spending Score],
  ...
]
```

---

# Step 4 – Apply the Elbow Method

```python
wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=0,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
```

### Explanation

```python
wcss = []
```

Creates an empty list to store WCSS values.

---

```python
for i in range(1, 11):
```

Tests:

```text
K = 1
K = 2
K = 3
...
K = 10
```

---

```python
n_clusters=i
```

Sets the current number of clusters.

---

```python
init="k-means++"
```

Uses the K-Means++ initialization strategy to select better initial centroids.

---

```python
random_state=0
```

Makes the result reproducible.

---

```python
n_init=10
```

Runs K-Means multiple times with different centroid initializations and keeps the better result.

---

```python
kmeans.fit(X)
```

Trains the K-Means model.

---

```python
kmeans.inertia_
```

Returns the WCSS value.

---

# Step 5 – Plot the Elbow Curve

```python
plt.plot(range(1, 11), wcss, marker="o")

plt.title("The Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.show()
```

The graph helps us determine an appropriate value of K.

---

# Step 6 – Create Final K-Means Model

After examining the elbow plot, we use:

```python
K = 5
```

Model:

```python
kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=0,
    n_init=10
)
```

---

# Step 7 – Fit and Predict

```python
y_kmeans = kmeans.fit_predict(X)
```

This performs two operations:

```text
fit       → learn the clusters
predict   → assign each customer to a cluster
```

The result might look like:

```text
[2, 1, 3, 0, 4, 2, 1, ...]
```

Each number represents the assigned cluster.

---

# Step 8 – Visualize Clusters

```python
plt.scatter(
    X[y_kmeans == 0, 0],
    X[y_kmeans == 0, 1],
    s=100,
    c="red",
    label="Cluster 1"
)

plt.scatter(
    X[y_kmeans == 1, 0],
    X[y_kmeans == 1, 1],
    s=100,
    c="blue",
    label="Cluster 2"
)

plt.scatter(
    X[y_kmeans == 2, 0],
    X[y_kmeans == 2, 1],
    s=100,
    c="green",
    label="Cluster 3"
)

plt.scatter(
    X[y_kmeans == 3, 0],
    X[y_kmeans == 3, 1],
    s=100,
    c="cyan",
    label="Cluster 4"
)

plt.scatter(
    X[y_kmeans == 4, 0],
    X[y_kmeans == 4, 1],
    s=100,
    c="magenta",
    label="Cluster 5"
)
```

Each color represents a different customer cluster.

---

# ⭐ Step 9 – Plot Centroids

```python
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="yellow",
    marker="X",
    label="Centroids"
)
```

`cluster_centers_` contains the coordinates of the cluster centroids.

For example:

```text
Cluster 1 → Centroid
Cluster 2 → Centroid
Cluster 3 → Centroid
Cluster 4 → Centroid
Cluster 5 → Centroid
```

---

# 📊 Complete Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# Load dataset
dataset = pd.read_csv("Mall_Customers.csv")


# Select Annual Income and Spending Score
X = dataset.iloc[:, [3, 4]].values


# --------------------------------
# Elbow Method
# --------------------------------

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=0,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)


# Plot Elbow Method

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.title("The Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.show()


# --------------------------------
# Final K-Means Model
# --------------------------------

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=0,
    n_init=10
)


# Assign customers to clusters

y_kmeans = kmeans.fit_predict(X)


# --------------------------------
# Visualize Clusters
# --------------------------------

plt.scatter(
    X[y_kmeans == 0, 0],
    X[y_kmeans == 0, 1],
    s=100,
    c="red",
    label="Cluster 1"
)

plt.scatter(
    X[y_kmeans == 1, 0],
    X[y_kmeans == 1, 1],
    s=100,
    c="blue",
    label="Cluster 2"
)

plt.scatter(
    X[y_kmeans == 2, 0],
    X[y_kmeans == 2, 1],
    s=100,
    c="green",
    label="Cluster 3"
)

plt.scatter(
    X[y_kmeans == 3, 0],
    X[y_kmeans == 3, 1],
    s=100,
    c="cyan",
    label="Cluster 4"
)

plt.scatter(
    X[y_kmeans == 4, 0],
    X[y_kmeans == 4, 1],
    s=100,
    c="magenta",
    label="Cluster 5"
)


# Plot centroids

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="yellow",
    marker="X",
    label="Centroids"
)


plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show()
```

---

# 🔢 Can We Use K = 6 or K = 10?

Yes.

For example:

### K = 6

```python
kmeans = KMeans(
    n_clusters=6,
    init="k-means++",
    random_state=0,
    n_init=10
)
```

This creates:

```text
6 customer segments
```

### K = 10

```python
kmeans = KMeans(
    n_clusters=10,
    init="k-means++",
    random_state=0,
    n_init=10
)
```

This creates:

```text
10 customer segments
```

However:

> More clusters does not automatically mean better clustering.

A suitable K should be selected using methods such as:

* Elbow Method
* Silhouette Score
* Domain/business knowledge
* Cluster interpretability

---

# 📏 Silhouette Score

Another method for evaluating clustering is the **Silhouette Score**.

It measures how well-separated the clusters are.

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, y_kmeans)

print("Silhouette Score:", score)
```

The score ranges approximately from:

```text
-1 → Poor separation
 0 → Overlapping clusters
+1 → Well-separated clusters
```

A higher silhouette score generally indicates better-defined clusters, but it should be considered alongside the business meaning of the segments.

---

# 📊 Standardization

K-Means is distance-based.

Therefore, feature scales can affect the clustering.

For example:

```text
Age                  → 18–70
Annual Income        → 15–150
Spending Score       → 1–100
```

When features have very different scales, standardization can be useful.

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Then:

```python
kmeans.fit_predict(X_scaled)
```

For this particular two-feature example, the features are already on fairly comparable ranges, but scaling becomes especially important when combining variables with very different units or magnitudes.

---

# ⚙️ Important K-Means Parameters

## `n_clusters`

Number of clusters.

```python
KMeans(n_clusters=5)
```

---

## `init`

Controls centroid initialization.

```python
init="k-means++"
```

Common choices include:

```text
"k-means++"
"random"
```

---

## `n_init`

Number of initializations to try.

```python
n_init=10
```

---

## `max_iter`

Maximum number of iterations for each initialization.

```python
max_iter=300
```

---

## `random_state`

Makes results reproducible.

```python
random_state=0
```

---

# 🏦 Real-World Applications

K-Means can be used in many areas.

### Customer Segmentation

```text
Customers
    ↓
K-Means
    ↓
Customer Groups
```

Useful for marketing and personalized campaigns.

### Banking

Possible applications include:

* Customer segmentation
* Transaction behavior grouping
* Product usage segmentation
* Risk behavior analysis
* Branch/customer analysis

### E-Commerce

* Customer segmentation
* Product grouping
* Purchasing behavior

### Computer Vision

K-Means can be used for:

* Image compression
* Color quantization
* Image segmentation

### NLP

K-Means can cluster:

* Documents
* Articles
* Customer reviews
* Text embeddings

---

# ⚠️ Limitations of K-Means

K-Means has several limitations.

### 1. Need to choose K

The number of clusters must be selected.

### 2. Sensitive to initialization

Different initial centroids can sometimes produce different results.

K-Means++ helps with initialization.

### 3. Sensitive to outliers

Extreme values can influence the centroid.

### 4. Works best with roughly spherical clusters

K-Means may perform poorly when clusters have complicated shapes.

### 5. Distance-based

Feature scaling can be important when variables have different scales.

---

# 🔄 K-Means vs Other Algorithms

| Algorithm               | Type         | Main Idea                    |
| ----------------------- | ------------ | ---------------------------- |
| K-Means                 | Unsupervised | Centroid-based clustering    |
| Hierarchical Clustering | Unsupervised | Builds hierarchy of clusters |
| DBSCAN                  | Unsupervised | Density-based clustering     |
| Logistic Regression     | Supervised   | Classification               |
| Decision Tree           | Supervised   | Rule-based prediction        |
| Random Forest           | Supervised   | Ensemble of decision trees   |

---

# 🧪 Project Workflow

```text
Mall Customers Dataset
          ↓
    Data Loading
          ↓
    Feature Selection
          ↓
Annual Income + Spending Score
          ↓
    Elbow Method
          ↓
      Select K
          ↓
    Train K-Means
          ↓
 Assign Customer Clusters
          ↓
 Visualize Centroids
          ↓
Customer Segmentation
```

---

# 💡 Key Learning Outcomes

Through this project, I learned:

* What unsupervised learning is
* How K-Means works
* How centroids are calculated
* How customers are assigned to clusters
* What WCSS means
* How to use the Elbow Method
* How to visualize clusters
* How `fit_predict()` works
* How K affects segmentation
* Why feature scaling can matter
* How clustering can be applied to business problems

---

# 🎤 Interview Questions

## Basic Questions

### 1. What is K-Means?

K-Means is an unsupervised machine learning algorithm that divides data into K clusters based on similarity.

### 2. Why is K-Means called unsupervised?

Because the dataset does not contain predefined labels or target values.

### 3. What does K represent?

K represents the number of clusters.

### 4. What is a centroid?

The centroid is the center point of a cluster.

### 5. How does K-Means assign data points?

It assigns each point to the nearest centroid based on distance.

---

## Intermediate Questions

### 6. What is WCSS?

WCSS is the sum of squared distances between data points and their assigned cluster centroids.

### 7. What is the Elbow Method?

It is a technique used to identify a suitable K by examining how WCSS decreases as K increases.

### 8. Why does WCSS decrease when K increases?

Because more clusters allow data points to be closer to their respective centroids.

### 9. What is K-Means++?

K-Means++ is an initialization method designed to select better starting centroids.

### 10. What does `fit_predict()` do?

It trains the model and returns the cluster assignment for each data point.

---

## Advanced Questions

### 11. Why is feature scaling important?

K-Means uses distance calculations, so features with larger numerical scales can have more influence.

### 12. What happens if K is too small?

Different naturally occurring groups may be combined into the same cluster.

### 13. What happens if K is too large?

A meaningful group may be divided into multiple smaller clusters.

### 14. What is the Silhouette Score?

It measures how well each point fits within its own cluster compared with neighboring clusters.

### 15. What are the limitations of K-Means?

Important limitations include:

* Need to select K
* Sensitivity to initialization
* Sensitivity to outliers
* Dependence on distance
* Less suitable for non-spherical cluster shapes

---

# 💼 Business Interpretation

The resulting customer clusters can be interpreted using:

```text
Annual Income
        +
Spending Score
        ↓
Customer Segment
```

For example, businesses may identify groups such as:

```text
High Income + High Spending
High Income + Low Spending
Low Income + High Spending
Low Income + Low Spending
Medium Income + Medium Spending
```

These segments can then support different marketing and customer-engagement strategies.

The exact interpretation should be based on the actual cluster centroids and business context rather than assuming every dataset will produce the same segments.

---

# 📁 Suggested Repository Structure

```text
K-Means-Clustering/
│
├── Mall_Customers.csv
├── kmeans_customer_segmentation.py
├── elbow_method.png
├── customer_clusters.png
└── README.md
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* K-Means Clustering

---

# 📦 Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

# ▶️ How to Run

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Navigate to the project:

```bash
cd K-Means-Clustering
```

Run:

```bash
python kmeans_customer_segmentation.py
```

The program will generate:

1. Elbow Method graph
2. Customer cluster visualization
3. Cluster centroids

---

# 📌 Conclusion

This project demonstrates how **K-Means Clustering** can be used to discover customer segments without predefined labels.

The project covers the complete workflow:

```text
Data
 ↓
Feature Selection
 ↓
Elbow Method
 ↓
K Selection
 ↓
K-Means
 ↓
Clusters
 ↓
Visualization
 ↓
Business Interpretation
```

The main takeaway is that **K-Means is not just about creating clusters**. The important part is selecting a suitable K, validating the clustering, and interpreting whether the resulting groups make sense for the real-world problem.

---

## ⭐ Project Highlights

```text
Algorithm       : K-Means Clustering
Learning Type   : Unsupervised Learning
Dataset         : Mall Customers
Features        : Annual Income + Spending Score
K Tested        : 1–10
Final K         : 5
Method          : Elbow Method
Evaluation      : WCSS / Silhouette Score
Visualization   : Matplotlib
```

---

## 👨‍💻 Author

**Subrata Mondal**

Data Analytics | Data Science | Machine Learning | Python | Power BI

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ and exploring the code.
