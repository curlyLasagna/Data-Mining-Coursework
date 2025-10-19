#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Assignment 1],
  authors: (
    (
      name: "Luis Dale Gascon",
      department: [Computer Science],
      organization: [Towson University],
      email: "lgascon1@students.towson.edu",
    ),
  ),
  index-terms: ("regression", "automobile", "cars"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction



= Dataset

The dataset is a modified version from Carnegie Mellon University's StatLib datasets. Each row contains information about a vehicle's horsepower, weight, acceleration, with an endgoal of predicting a vehicle's miles per gallon

= Exploratory Data Analysis

== Missing Values

There are a total of 6 rows that were missing values for their horsepower feature. Since we're working with 398 points, we've decided to simply remove those 6 points as they only account for about 1% of the data.


= Data Preprocessing

Out of the 398 rows in the dataset, 6 rows contain missing information on horsepower. Those rows accounts about ~1% of the dataset, so we've decided to drop those rows.

== Data Binning
We bin the data with clustering-based binnning by using the k-means clustering algorithm to group the data based on similarities.

= Regression

= Conclusion
