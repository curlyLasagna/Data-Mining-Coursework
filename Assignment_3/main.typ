#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Assignment 3],
  authors: (
    (
      name: "Luis Dale Gascon",
      department: [Computer Science],
      organization: [Towson University],
      email: "lgascon1@students.towson.edu",
    ),
  ),
  index-terms: ("coffee", "clustering", "unsupervised learning"),
  figure-supplement: [Fig.],
  abstract: [
    This paper investigates the relationship between the geographic origin of coffee and its sensory characteristics by applying unsupervised clustering methods to a dataset of coffee evaluations. The dataset includes features such as aroma, flavor, aftertaste, acidity, and balance, along with the country of origin for each sample. We compare the effectiveness of three clustering algorithms DBSCAN, Agglomerative Clustering, and Mean Shift in grouping coffees based on their sensory profiles. Our analysis aims to determine whether coffees from the same region exhibit similar taste profiles and to what extent geography influences sensory perception. The results provide insights into the role of origin in shaping coffee quality and inform both producers and consumers about the potential of clustering techniques in coffee analysis.
  ],
)

= Introduction
The sensory qualities of coffee are shaped by environmental, genetic, and processing factors, with geographic origin often cited as a key influence. Understanding how origin relates to taste profiles can better inform consumers about their coffee preferences. In this study, we apply unsupervised clustering techniques to a dataset of coffee evaluations, aiming to uncover patterns in sensory attributes linked to geographic regions. By comparing the performance of DBSCAN, Agglomerative Clustering, and Mean Shift algorithms, we look to determine whether coffees from similar origins exhibit comparable sensory characteristics, and to assess the effectiveness of clustering methods in analyzing coffee quality.

= Dataset

The dataset contains 41 columns, 14 of which we decided to drop. The columns that we decided to drop are

#columns(2)[
  - `ID`
  - `Defects`
  - `Status`
  - `Farm Name`
  - `Lot Number`
  - `Mill`
  - `ICO Number`
  #colbreak()
  - `Company`
  - `Producer`
  - `Status`
  - `Certification Body`,
  - `Certification Address`,
  - `Certification Contact`
]
The columns listed above is considered noise with what we're interested in.
The dataset is fairly small at 207 rows in total.

= Data Analysis

Focusing on the features relating to geography, we want to analyze altitude and country of origin.

#image("country_dist.png")

The majority of the datapoints are from Taiwan. Taiwan has a humid and tropical climate with seasonal monsoons.

#image("altitude_dist.png")

#image("non_cluster_scatter.png")

Reducing the dataset down to 2 components via PCA shows that the dataset is already well clustered.

= Background

Before we implement the clustering algorithms listed above, we need to provide some background information about each algorithm.

The clustering algorithms tested in this experiment do not require an initial number of clusters, great if you don't have a solid understanding of the dataset.

== DBSCAN

A density-based clustering method that groups together points that are closely packed.
Requires 2 hyperparamters: $epsilon$ and $"minPts"$. $epsilon$ defines the radius of the neighborhood around a single data point

Having a good background about the data is important in choosing a meaningful value for $epsilon$. If $epsilon$ is too small, then most points will be labeled as noise. On the other hand, if $epsilon$ is too large, then it may result in a large cluster instead of a well separated one.

DBSCAN creates clusters by first identifying the core points through counting the number of points within its $epsilon$. If the the count of points meets minimum points, then it's a core point. Once the core points are identified, recursively find points that are within the $epsilon$ of the core points and add those to the cluster. Points considered noise are any points that doesn't belong to a cluster.

== Mean-Shift

Similar to DBSCAN, it doesn't assume a predefined shape for clusters, which allows it to handle clusters of arbitrary shapes and is a density-based algorithm.

Also called a mode-seeking algorithm since it seeks the most dense region of points.
It's hyperparameter is its bandwidth. Bandwidth determines the size of the neighborhood around each data point. Conveniently, sklearn's MeanShift implementation uses a function that estimates the optimal bandwidth by default.

The algorithm begins with each data point looking for other points within its bandwidth distance. It calculates the mean of all the other data points within its bandwidth and that mean is where its the most dense. This process repeats until the mean no longer moves, also called convergance. The point where the points converge becomes the cluster center.


== Agglomerative Clustering

The only distance-based clustering algorithm that we're testing.

Agglomerative clustering is a bottom-up approach to hierarchical clustering where each data point starts with its own cluster then the data points merged together to form a bigger cluster. Clusters are formed by calculating the distance between every pair of clusters and merging the closest two clusters. The distances between exsiting clusters are recalculated and this process repeats until a predefined number of cluster or distance threshold is achieved.

sklearn's implementation of agglomerative clustering uses euclidean as a distance metric by default and 2 as the number of clusters to find.

= Methodology

We want to explore the 3 different clustering algorithms mentioned in the previous section for our analysis.

We explicitly set $epsilon$ of 2 and minimum points of 5 for the DBSCAN algorithm due to how well clustered the data already is as shown in the data analysis section. For the other two algorithms, we opt to use their default parameters provided by sklearn.

To evaluate the performance of each clustering algorithm, we will be calculating the silhoutte score of each algorithm.

== Preprocessing

Clustering algorithms are sensitive to the scale of the features as most algorithms are distance-based. Differences in scales could disproportionately influence the distance calculation. To address this issue, we set the numerical features of the data to a standard scale.

Region, a categorical feature, has high cardinality of 121, so we opted to use frequency encoding to avoid adding too much features. As for the rest of the categorical features, we encode those features via one hot encoding as the rest do not have a high cardinality as region.

= Results

Agglomerative clustering produced 2 clusters. Mean Shift produced 4 clusters and DBSCAN produced 3 clusters and gave noise a cluster label of -1.

We initially didn't consider applying dimensionality reduction in our preprocessing step, but after reading about the curse of dimensionality, where data becomes increasingly sparse due to high number of features, we decided to compare the clusters with and without reducing dimensions.

#image("cum_variance.png")

To get the most optimal number of clusters, we have two metrics we could analyze. The first metric would be to evaluate the cumulative explained variance.

There are a total of 12 features in the dataset. The cumulative explained variance plot shows that at 5 components, we would retain 90% of variance of the dataset, which is known to be best practice.

#image("scree_plot.png")

According to the scree plot, the elbow point is at 3 components, which is where the drop in variance gains dramatically flattens out.


#table(
  columns: (auto, auto, auto),
  table.header([*Algorithm*], [Silhoutte score], [Silhoutte score with PCA]),
  [Mean Shift], [0.43], [0.56],
  [DBSCAN], [0.57], [0.57],
  [Agglomerative], [0.54], [0.54],
)

The silhoutte score didn't change much after reducing dimenions except for the mean shift algorithm. These results suggests that PCA captured the most relevant information for clustering.

= Evaluating the Results

Using Altair as our choice of interactive visualization tool, we were able to analyze the clusters. We reduced the dimension of the results down to 2 dimensions using PCA to plot a 2d scatter plot with clusters as their color.

#image("meanshift_clusters.png")
#image("DBSCAN_clusters.png")

Interestingly, the noise labeled by DBSCAN has interestesting properties. The noise on the left of the scatter plot are country's with altitude as their outliers as they have really high values.
#image("agglomerative_clusters.png")

The cluster of data points on the right primarily represents samples from Taiwan. The central cluster consists of a mix of samples from Thailand and Taiwan, while the clusters on the left correspond to various Latin American countries.

If the dataset had included latitude information for the regions, it is likely that the clustering results would have been even more distinct. Given that there are over 100 unique regions, manually searching their respective latitude would be a monumental tasks.

= Conclusion

This study explored the relationship between the geographic origin of coffee and its sensory characteristics using unsupervised clustering algorithms. By applying DBSCAN, Agglomerative Clustering, and Mean Shift to a dataset of coffee evaluations, we found that samples from similar regions often grouped together. Dimensionality reduction with PCA only improved clustering performance for one algorithm, highlighting the importance of feature selection. While the results suggest a clear link between origin and sensory profile, the analysis was limited by the lack of detailed geographic data such as latitude. Future work could be sourcing a larger dataset with more location information. Overall, clustering techniques show promise for uncovering patterns in coffee quality and can aid coffee enthusiast in understanding the influence of origin on flavor.
