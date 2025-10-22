import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import numpy as np
    from ucimlrepo import fetch_ucirepo
    from sklearn.preprocessing import (
        OrdinalEncoder,
        StandardScaler,
        MinMaxScaler,
        FunctionTransformer,
        KBinsDiscretizer,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LinearRegression
    from scipy.stats import skew, probplot, kurtosis, shapiro
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    return (
        FunctionTransformer,
        KBinsDiscretizer,
        KMeans,
        LinearRegression,
        OrdinalEncoder,
        StandardScaler,
        alt,
        fetch_ucirepo,
        make_column_transformer,
        make_pipeline,
        mean_absolute_error,
        mo,
        np,
        pd,
        probplot,
        root_mean_squared_error,
        shapiro,
        train_test_split,
    )


@app.cell
def _(fetch_ucirepo):
    # fetch dataset
    auto_mpg = fetch_ucirepo(id=9)

    # data (as pandas dataframes)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    # Drop rows with null values
    X = X.dropna()
    y = y.loc[X.index]
    return X, y


@app.cell
def _(X, train_test_split, y):
    # Split test and train datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train):
    X_train
    return


@app.cell
def _(mo):
    mo.md(r"""## EDA""")
    return


@app.cell
def _(KMeans, alt, pd, probplot, y):
    def simple_bar_chart(df: pd.DataFrame, **kwargs: str) -> alt.Chart:
        """
        Creates a simple bar chart of a single categorical column.

        Args:
            df: The pandas DataFrame.
            **kwargs: Must contain 'column' with the name of the categorical column.

        Returns:
            An Altair Chart object.
        """
        column = kwargs.get("col")
        if not column:
            raise ValueError("A 'column' keyword argument must be provided.")

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(column, title=column),
                y=alt.Y("count()", title="Count"),
                tooltip=[column, "count()"],
            )
            .properties(title=f"Bar Chart of {column}")
        )

        kde = (
            alt.Chart(df)
            .transform_density(
                density=column,
                as_=[column, "density"],
            )
            .mark_line(strokeWidth=3, color="darkred")
            .encode(
                x=alt.X(column, title=column),
                y=alt.Y("density:Q", axis=None),
                tooltip=[column, "density:Q"],
            )
        )

        return alt.layer(chart, kde).resolve_scale(y="independent")


    def scatter_plot(df: pd.DataFrame, **kwargs: str) -> alt.Chart:
        column = kwargs.get("col")
        target_var = kwargs.get("y")
        if not column:
            raise ValueError("A 'column' keyword argument must be provided.")

        chart = (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X(column, title=column),
                y=alt.Y(target_var, title=y),
                tooltip=[column, target_var],
            )
            .properties(title=f"{column} x {target_var}", width=1000)
        )
        return chart


    def qq_plot(data, title: str) -> alt.Chart:
        (osm, osr), (slope, intercept, r) = probplot(
            pd.Series(data), dist="norm", plot=None
        )

        qq_df = pd.DataFrame(
            {"Theoretical Quantiles (Z-Score)": osm, "Sample Quantiles": osr}
        )

        line_df = pd.DataFrame(
            {
                "Theoretical Quantiles (Z-Score)": [min(osm), max(osm)],
                "Reference Line": [
                    min(osm) * slope + intercept,
                    max(osm) * slope + intercept,
                ],
            }
        )

        scatter = (
            alt.Chart(qq_df)
            .mark_circle(size=60, color="#10B981")
            .encode(
                x=alt.X(
                    "Theoretical Quantiles (Z-Score)", axis=alt.Axis(grid=True)
                ),
                y=alt.Y("Sample Quantiles"),
                tooltip=["Theoretical Quantiles (Z-Score)", "Sample Quantiles"],
            )
        )

        line = (
            alt.Chart(line_df)
            .mark_line(color="#EF4444", strokeWidth=3)
            .encode(x="Theoretical Quantiles (Z-Score)", y="Reference Line")
        )

        chart = (scatter + line).properties(title=title).interactive()

        return chart


    def elbow_chart(df: pd.DataFrame, k_range) -> alt.Chart:
        wcss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k).fit(df[["weight"]])
            wcss.append(kmeans.inertia_)

        elbow_df = pd.DataFrame({"K": k_range, "Inertia": wcss})

        chart = (
            alt.Chart(elbow_df)
            .encode(
                x=alt.X("K", type="ordinal"),
                y=alt.Y("Inertia"),
                tooltip=["K", "Inertia"],
            )
            .properties(width="container")
        )
        line = chart.mark_line(point=True)

        points = chart.mark_point(size=60, filled=True, color="red")
        elbow_point = elbow_df[elbow_df["K"] == 5]
        return (line + points).interactive()
    return elbow_chart, qq_plot, scatter_plot, simple_bar_chart


@app.cell
def _(X_train, elbow_chart):
    elbow_chart(X_train, range(2, 15)).save(fp="elbow_weight.png", scale_factor=2)
    return


@app.cell
def _(X, X_train, simple_bar_chart):
    for X_col in X_train.select_dtypes(include=["float"]).columns.to_list():
        simple_bar_chart(X, col=X_col).show()
    return


@app.cell
def _(X_train, simple_bar_chart):
    simple_bar_chart(X_train, col="displacement").save(
        "displacement_bar.png", scale_factor=2
    )
    return


@app.cell
def _(X, scatter_plot, y):
    scatter_plot(X.join(y), col="horsepower", y="mpg").save(
        fp="horsepower_x_mpg.png", scale_factor=2
    )
    return


@app.cell
def _(X, scatter_plot, y):
    scatter_plot(X.join(y), col="acceleration", y="mpg")
    return


@app.cell
def _(X, scatter_plot, y):
    scatter_plot(X.join(y), col="model_year", y="mpg").save(
        fp="model_year_x_mpg.png", scale_factor=2
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Normal distribution""")
    return


@app.cell
def _(FunctionTransformer, np):
    log_transformer = FunctionTransformer(np.log, feature_names_out="one-to-one")
    sqrt_transformer = FunctionTransformer(np.sqrt, feature_names_out="one-to-one")
    inverse_sqrt_transformer = FunctionTransformer(
        lambda n: 1 / np.sqrt(n), feature_names_out="one-to-one"
    )
    return inverse_sqrt_transformer, log_transformer, sqrt_transformer


@app.cell
def _(mo):
    mo.md(r"""## Encoding""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Label Encoding

    For features with low-cardinality, we want to apply ordinal encoding

    - cylinders
    - model_year
    """
    )
    return


@app.cell
def _(X_train):
    X_train.select_dtypes(include=["float"]).columns.to_list()
    return


@app.cell
def _(
    KBinsDiscretizer,
    OrdinalEncoder,
    StandardScaler,
    X_train,
    inverse_sqrt_transformer,
    log_transformer,
    make_column_transformer,
    make_pipeline,
    sqrt_transformer,
):
    # Rescaling pipelines
    numerical_pipeline = make_pipeline(StandardScaler())
    categorical_pipeline = make_pipeline(OrdinalEncoder())

    # Discretization pipelines
    kbins_uniform_pipeline = make_pipeline(
        KBinsDiscretizer(n_bins=9, strategy="uniform", encode='ordinal')
    )
    kbins_kmeans_pipeline = make_pipeline(
        KBinsDiscretizer(n_bins=4, strategy="kmeans")
    )

    # Normality pipelines
    displacement_log = make_pipeline(log_transformer)
    displacement_sqrt = make_pipeline(sqrt_transformer)
    displacement_inverse_sqrt = make_pipeline(inverse_sqrt_transformer)

    numerical_cols = X_train.select_dtypes(include=["float"]).columns.to_list()
    categorical_cols = ["cylinders", "model_year", "origin"]

    pre_processor = make_column_transformer(
        (numerical_pipeline, numerical_cols),
        (categorical_pipeline, categorical_cols),
        (kbins_kmeans_pipeline, ["weight"]),
        (displacement_log, ["displacement"]),
    )
    return (
        displacement_inverse_sqrt,
        displacement_log,
        displacement_sqrt,
        pre_processor,
    )


@app.cell
def _(X_test, X_train, pd, pre_processor):
    pre_processed_train = pd.DataFrame(
        data=pre_processor.fit_transform(X=X_train),
        columns=pre_processor.get_feature_names_out(),
        index=X_train.index,
    )

    pre_processed_test = pd.DataFrame(
        data=pre_processor.fit_transform(X=X_test),
        columns=pre_processor.get_feature_names_out(),
        index=X_test.index,
    )
    return pre_processed_test, pre_processed_train


@app.function
def shapiro_result(res, trans):
    return (
        print(f"{trans} Sample looks Gaussian (fail to reject H0)")
        if res.pvalue > 0.05
        else print(f"{trans} Sample does not look Gaussian (reject H0)")
    )


@app.cell
def _(
    X,
    displacement_inverse_sqrt,
    displacement_log,
    displacement_sqrt,
    make_column_transformer,
    qq_plot,
    shapiro,
):
    for k, v in {
        "log": displacement_log,
        "sqrt": displacement_sqrt,
        "in_sqrt": displacement_inverse_sqrt,
    }.items():
        trans_arr = make_column_transformer((v, ["displacement"])).fit_transform(X)

        shapiro_result(res=shapiro(trans_arr), trans=k)

        qq_plot(data=trans_arr.flatten(), title=k).show()
        # .save(f"{k}_transform.png", scale_factor=2)
    return


@app.cell
def _(
    LinearRegression,
    mean_absolute_error,
    pre_processed_test,
    pre_processed_train,
    root_mean_squared_error,
    y_test,
    y_train,
):
    def training_metrics():
        le = LinearRegression().fit(X=pre_processed_train, y=y_train)
        y_pred = le.predict(pre_processed_test)
        r2 = le.score(pre_processed_test, y_test)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        rmse = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
        return {"r2": r2, "mae": mae, "rmse": rmse}
    return (training_metrics,)


@app.cell
def _(training_metrics):
    training_metrics()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
