import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import numpy as np
    from ucimlrepo import fetch_ucirepo
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LinearRegression
    return (
        KMeans,
        OrdinalEncoder,
        StandardScaler,
        alt,
        fetch_ucirepo,
        make_column_transformer,
        make_pipeline,
        mo,
        pd,
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
    return (X_train,)


@app.cell
def _(mo):
    mo.md(r"""## EDA""")
    return


@app.cell
def _(KMeans, alt, pd, y):
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
        return chart


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
            .properties(title=f"{column} x {target_var}")
        )
        return chart


    def elbow_chart(df: pd.DataFrame, k_range) -> alt.Chart:
        wcss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k).fit(df)
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
    return elbow_chart, scatter_plot, simple_bar_chart


@app.cell
def _(X, elbow_chart):
    elbow_chart(X, range(1, 10))
    return


@app.cell
def _(X, simple_bar_chart):
    simple_bar_chart(X, col="horsepower")
    return


@app.cell
def _(X, scatter_plot, y):
    scatter_plot(X.join(y), col="horsepower", y="mpg")
    return


@app.cell
def _(X, scatter_plot, y):
    scatter_plot(X.join(y), col="acceleration", y="mpg")
    return


@app.cell
def _(X, scatter_plot, y):
    scatter_plot(X.join(y), col="model_year", y="mpg")
    return


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
def _(
    OrdinalEncoder,
    StandardScaler,
    X_train,
    make_column_transformer,
    make_pipeline,
):
    numerical_pipeline = make_pipeline(StandardScaler())
    categorical_pipeline = make_pipeline(OrdinalEncoder())

    numerical_cols = X_train.select_dtypes(include=["float"]).columns.to_list() + [
        "weight"
    ]
    categorical_cols = ["cylinders", "model_year", "origin"]

    pre_processor = make_column_transformer(
        (numerical_pipeline, numerical_cols), (categorical_pipeline, categorical_cols)
    )
    return (pre_processor,)


@app.cell
def _(X_train, pd, pre_processor):
    pre_processed_df = pd.DataFrame(
        data=pre_processor.fit_transform(X=X_train),
        columns=pre_processor.get_feature_names_out(),
        index=X_train.index
    )
    return (pre_processed_df,)


@app.cell
def _(pre_processed_df):
    pre_processed_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
