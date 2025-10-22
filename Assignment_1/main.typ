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
    // bibliography: bibliography("refs.bib"),
    figure-supplement: [Fig.],
)

= Introduction

Gasoline is both expensive and environmentally damaging, making fuel efficiency an important concern for consumers and manufacturers. Understanding how vehicle features such as horsepower, weight, acceleration, and displacement can influence miles per gallon (mpg) will allow for informed decision-making and promote sustainability. This paper aims to identify the key predictors of mpg by constructing a linear regression model while experimenting with different pre-processing methods and evaluating its performance using established metrics.

= Dataset

The dataset is a modified version from Carnegie Mellon University's StatLib dataset. Each row contains the following vehicle information:

- `horsepower`: Enging power output
- `weight`: Vehicle mass
- `acceleration`: Rate at which velocity changes over time
- `displacement`: Total volume of air and fuel an engine can displace
- `model_year`: A vehicle's production period

These features are used to predict a vehicle's miles per gallong (mpg), which is the focus of our regression model and is the prediction question.

We split the training and testing dataset via 70/30 where we take 70% of the dataset as the training data and 30% as the testing data to avoid an overfitted model.

= Exploratory Data Analysis

== Missing Values

6 rows from the dataset were missing horsepower values (about 1% of 398 total rows). We opted to remove these rows to maintain data quality with minimal loss.

== Scatter Plots

Scatter plots were generated to explore relationships between the target variable: mpg and individual features to visualize their patterns.

#figure(
    image("./horsepower_x_mpg.png"),
    caption: [Mpg decreases as horsepower increasese]
)

#figure(
    image("./model_year_x_mpg.png"),
    caption: [Mpg increase as model year increases]
)


= Data Preprocessing

== Normalizing numerical features

We want to set a standard value for numerical values to allow features with different magnitudes to contribute the same impact. I took features of type float such as displacement, horsepower and acceleration to get standardized.

== Data Binning

We plan to experiment with 2 binning techniques: Equal Width and K-Means Clustering

For the inital binning technique, we want to find the optimal number of bins via Sturges' Rule: $ceil(log_(2)n + 1)$, where $n$ is the height of our dataset. Our training set has 274 rows, so we plug that value into Sturges' rule and we get a value of 9.

For the second binning technique. We tested a range of clusters from 2 to 10. We plotted each cluster's Within Cluster Sum of Squares or WCSS. The equation for WCSS is as follows:


$sum_(i=1)^k sum_(x in C_i) || x - mu_i||^2$

#figure(
    image("./elbow_weight.png", width: 70%),
    caption: [Elbow plot]

)

Looking at the plot, we can estimate that 4 would be the optimal value of clusters to set for K-means. We utilized Sklearn's `KBinsDiscretizer` to perform both types of binning by setting the keyword argument `strategy` to `uniform` or `kmeans`.

== Transforming a feature with non-normal distribution

Looking at the histogram for the feature of displacement, the data is positively skewed. To mitigate the impact of outliers and improve the performance of our regression model, we apply 3 different kinds of data transformation independently to compare their results.

I applied natural log, square root and inverse square root transformations to the dataset invidually. To evaluate how each transformation affected normality, I used a normal Quantile-Quantil (QQ) plot to visually compare the quantiles of the transformed data to those of a normal distribution.

#figure(
    image("./displacement_bar.png", width: 70%),
    caption: [Displacement shape via kernel density estimation]
)

#figure(
    image("./log_transform.png", width: 70%),
    caption: [Log transformation normal QQ plot]
)

#figure(
    image("./sqrt_transform.png", width: 70%),
    caption: [Square root transformation normal QQ plot]
)


#figure(
    image("./in_sqrt_transform.png", width: 70%),
    caption: [Inverse square root transformation normal QQ plot]
)



Visually, the normal QQ plot from a log transformation is the closest to matching the normal distribution, so for our data pre-processing step, we'll apply log transformation to the displacement feature.

= Regression Analysis

To evaluate our model, we look at 3 metrics: $R^2$ score, and mean absolute error (MAE)

#align(center,
    figure(
        table(
            columns: (auto, auto),
            table.header([Metric], [Score]),
            $R^2$, $0.81$,
            "MAE", $2.56$,
        ),
    )
)

$R^2$ score, also known as coefficient of determination is a measure of how predictable the target variable is based on the features of the dataset. We can say that $81%$ of the variation of the target variable, mpg, can be explained by the features of the training data. As for MAE, on average, the model's prediction are off by 2.56 mpg.

We wanted to see if the binning method chosen made a difference in the model's performance, and it did. Equal width binning gave us the $R^2$ score that we see in the table, while binning by clustering gave us an $R^2$ score of $.79$

= Conclusion

Our analysis shows how a vehicle's features are significant predictors of miles per gallon. By experimenting with different data preprocessing methodologies and constructing a linear regression model, we achieved a model with a respectable $R^2$ score and low mean absolute error, which indicates
