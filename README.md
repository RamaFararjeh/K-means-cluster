# PRODIGY-TrackCode-02

🛍 I recently had the opportunity to create a K-means clustering algorithm to group customers based on their purchase history. Here's a brief overview of the process and some key insights: 🛒💳

-Data Exploration:
I started by loading the customer data and exploring its basic statistics. The dataset contains information about 200 customers.

-Data Preprocessing:

-Checked for missing values and found that there were no missing entries in the dataset.
Transformed categorical features like 'Gender' into numerical values using Label Encoding for compatibility with the K-means algorithm.
Elbow Method for K Selection:
I used the "Elbow Method" to determine the optimal number of clusters (K). The plot of "Within-Cluster Sum of Squares (WCSS)" against K showed an "elbow" point where the rate of decrease in WCSS slowed down. I chose K=5 clusters based on this analysis.

-K-means Clustering:

Created a K-means model with K=5 clusters.
Fit the model to the data and assigned each customer to one of the five clusters.

Feel free to reach out if you have any questions or want to delve deeper into this exciting project! 👩‍💼🛒📊
#DataScience #Retail #CustomerSegmentation #KMeansClustering #MachineLearning

