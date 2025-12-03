import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import kagglehub
    import pandas as pd
    import altair as alt
    from pathlib import Path
    from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering, estimate_bandwidth
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline, FunctionTransformer
    from sklearn.compose import ColumnTransformer, make_column_transformer
    from sklearn.decomposition import PCA
    return (
        AgglomerativeClustering,
        DBSCAN,
        MeanShift,
        OneHotEncoder,
        Path,
        StandardScaler,
        kagglehub,
        make_column_transformer,
        make_pipeline,
        pd,
    )


@app.cell
def _(Path, kagglehub, pd):
    # Download latest version
    path = kagglehub.dataset_download("fatihb/coffee-quality-data-cqi")
    df = pd.read_csv(Path(path) / "df_arabica_clean.csv").drop(
        columns=[
            "Unnamed: 0",
            "Defects",
            "ICO Number",
            "Farm Name",
            "Lot Number",
            "Mill",
            "ICO Number",
            "Company",
            "Producer",
            "Status",
            "Variety",
            "Certification Body",
            "Certification Address",
            "Certification Contact",
            "Grading Date",
            "In-Country Partner",
            "Owner",
            "Processing Method",
            "Bag Weight",
            "Number of Bags",
            "Harvest Year",
            "Total Cup Points",
            "Moisture Percentage",
            "Category One Defects",
            "Quakers",
            "Category Two Defects",
            "Expiration",
        ]
    )
    return (df,)


@app.cell
def _(np, pd):
    def fixup_data(df: pd.DataFrame):
        def clean_altitude_range(range_value):
            if isinstance(range_value, str):
                # Remove blank spaces
                range_value = range_value.replace(" ", "")
                if "-" in range_value:
                    try:
                        start, end = range_value.split("-")
                        start = int(start)
                        end = int(end)
                        return (start + end) / 2
                    except ValueError:
                        return np.nan
                else:
                    try:
                        return int(range_value)
                    except ValueError:
                        return np.nan
            else:
                return range_value
            
        fixed_df = df.copy(deep=True)
        # Missing values
        fixed_df.loc[df["ID"] == 99, "Altitude"] = 5273  # Impute value for ID 99
        fixed_df.loc[df["ID"] == 105, "Altitude"] = 1800  # Impute value for ID 105
        fixed_df.loc[df["ID"] == 180, "Altitude"] = 1400  # Impute value for ID 180
        fixed_df["Altitude"] = fixed_df["Altitude"].apply(clean_altitude_range)
        fixed_df.fillna({"Region": "Uknown"}, inplace=True)

        return fixed_df
    return (fixup_data,)


@app.cell
def _(df, fixup_data):
    fixup_data(df)
    return


@app.cell
def _(df):
    df["Country of Origin"].unique()
    return


@app.cell
def _(df):
    df_country_region = (
        df.groupby("Country of Origin")["Region"].agg(list).reset_index()
    )
    df_country_region["Region_Count"] = df_country_region["Region"].apply(len)
    return (df_country_region,)


@app.cell
def _(df_country_region):
    df_country_region
    return


@app.cell
def _(fixed_df):
    fixed_df
    return


@app.cell
def _(
    OneHotEncoder,
    StandardScaler,
    df,
    fixup_data,
    make_column_transformer,
    make_pipeline,
    pd,
):
    fixed_df = fixup_data(df).drop(columns=["ID"])
    numerical_pipeline = make_pipeline(StandardScaler())
    categorical_pipeline = make_pipeline(OneHotEncoder(sparse_output=False))
    numerical_col = fixed_df.select_dtypes(include=["float64"]).columns
    categorical_col = fixed_df.select_dtypes(include=["object"]).columns

    pre_processor = make_column_transformer(
        (numerical_pipeline, numerical_col),
        (categorical_pipeline, categorical_col),
        remainder="passthrough"
    )

    preprocessed_df = pd.DataFrame(
        data=pre_processor.fit_transform(fixed_df),
        columns=pre_processor.get_feature_names_out(),
        index=fixed_df.index
    )
    return fixed_df, preprocessed_df


@app.cell
def _(pd):
    def label_data(clustering, df: pd.DataFrame) -> pd.DataFrame:
        labeld_df = df.copy(deep=True)
        labeld_df["Cluster"] = clustering.labels_
        return labeld_df
    return (label_data,)


@app.cell
def _(AgglomerativeClustering, df, label_data, preprocessed_df):
    label_data(AgglomerativeClustering().fit(preprocessed_df), df)
    return


@app.cell
def _(DBSCAN, df, label_data, preprocessed_df):
    label_data(DBSCAN(eps=2.5, min_samples=2).fit(preprocessed_df), df)
    return


@app.cell
def _(MeanShift, df, label_data, preprocessed_df):
    label_data(MeanShift().fit(preprocessed_df), df)
    return


app._unparsable_cell(
    r"""
    def visualize_clusters():
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
