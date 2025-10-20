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

The dataset is a modified version from Carnegie Mellon University's StatLib dataset. Each row contains the following information about a vehicle:

- `horsepower`: Describes the power output of engines
- `weight`: Describes how heavy something is
- `acceleration`: Rate at which velocity changes over time
- `displacement`: Total volume of air and fuel an engine can displace
- `model_year`: A vehicle's production period

These features are stated to have an endgoal of predicting a vehicle's miles per gallon, which is the prediction question that I'll work with to create a regression model.

= Exploratory Data Analysis

== Missing Values

There are a total of 6 rows that were missing values for their horsepower feature. Since we're working with 398 points, we've decided to simply remove those 6 points as they only account for about 1% of the data.

== A feature with non-normal distribution

Looking at the histogram for the feature of displacement, the data is positevly skewed.

#image("./displacement_bar.png")

Applying a log transformation

#image("./normalized_displacement.png")

Using inverse square root, we get a skewness of 0.7846490216580512
log transformation = -0.4154140125333754
square root = -0.07403252121663605

Getting the absolute value of our results, square root transformation comes out on top

= Data Preprocessing

We want to set a standard value for numerical values to allow features with different magnitudes to contribute the same impact. I took features of type float such as displacement, horsepower and acceleration. I also included weight.

== Data Binning
We bin the data with clustering-based binnning by using the k-means clustering algorithm to group the data based on similarities.

= Regression



= Conclusion
