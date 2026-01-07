import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import kagglehub
    import numpy as np
    import pandas as pd
    import altair as alt
    from pathlib import Path
    from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering, estimate_bandwidth
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from category_encoders.count import CountEncoder
    from sklearn.pipeline import make_pipeline, FunctionTransformer
    from sklearn.compose import ColumnTransformer, make_column_transformer
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    return (
        AgglomerativeClustering,
        CountEncoder,
        DBSCAN,
        MeanShift,
        OneHotEncoder,
        PCA,
        Path,
        StandardScaler,
        alt,
        kagglehub,
        make_column_transformer,
        make_pipeline,
        np,
        pd,
        silhouette_score,
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
            "Sweetness",
            "Clean Cup"
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
def _(alt, df, fixup_data, pd):
    def create_altitude_bar_chart(df: pd.DataFrame, filename: str = None) -> alt.Chart:
        # 1. Base Chart with a title
        base = alt.Chart(df).properties(
            title='Distribution of Altitude'
        )

        # 2. Histogram (Bar Chart)
        chart = base.mark_bar().encode(
            # X-axis: 'Altitude' binned to show distribution. maxbins=20 is a good default.
            x=alt.X('Altitude', bin=alt.Bin(maxbins=20), title='Altitude (meters)'),
            # Y-axis: Count of records in each bin
            y=alt.Y('count()', title='Number of Samples'),
            # Tooltip for interactivity
            tooltip=[alt.Tooltip('Altitude', bin=True), alt.Tooltip('count()', title='Number of Samples')]
        ).interactive() # Allows for zooming and panning

        return chart

    create_altitude_bar_chart(fixup_data(df)).save(fp="altitude_dist.png", scale_factor=2)
    return


@app.cell
def _(alt, fixed_df, pd):
    def create_taste_feature_distribution_chart(df: pd.DataFrame) -> alt.Chart:
        """
        Returns an Altair chart showing the distribution of the taste features 
        (Aroma, Flavor, Aftertaste, Acidity, Body, Balance) of the coffee dataset.
        """
        taste_features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
    
        df_long = df[taste_features].melt(
            var_name='Taste Feature',
            value_name='Score'
        )

        base = alt.Chart(df_long).properties(
            title='Distribution of Coffee Taste Features'
        )

        chart = base.transform_density(
            density='Score',
            groupby=['Taste Feature'],
            # FIX: Renaming output score to Score_KDE for clarity
            as_=['Score_KDE', 'Density'], 
            extent=[df_long['Score'].min() - 0.5, df_long['Score'].max() + 0.5] 
        ).mark_area(
            opacity=0.6,
            interpolate='monotone', 
        ).encode(
            # FIX: Using Score_KDE
            x=alt.X('Score_KDE:Q', title='Taste Score', scale=alt.Scale(domain=[6.5, 9.0])), 
            # FIX: Explicitly specifying type='quantitative' for the transformed field
            y=alt.Y('Density', type='quantitative', title='Density (KDE)'),
            color=alt.Color('Taste Feature:N', legend=alt.Legend(title="Feature")),
            tooltip=['Taste Feature', alt.Tooltip('Score', format='.2f'), alt.Tooltip('Density', format='.2f')]
        ).interactive()

        return chart

    create_taste_feature_distribution_chart(fixed_df)
    return


@app.cell
def _(fixed_df):
    taste_features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
    
    df_long = fixed_df[taste_features].melt(
        var_name='Taste Feature',
        value_name='Score'
    )
    df_long
    return


@app.cell
def _(fixed_df):
    fixed_df
    return


@app.cell
def _(fixed_df):
    df_country_region = (
       fixed_df 
        .groupby("Country of Origin")["Region"]
        .agg(list)
        .reset_index()
    )
    df_country_region["Region_Count"] = df_country_region["Region"].apply(len)
    return (df_country_region,)


@app.cell
def _(alt, df_country_region, pd):
    def create_country_region_altair_chart(df: pd.DataFrame, filename: str = None) -> alt.Chart:
        chart = alt.Chart(df).mark_bar().encode(
            # X-axis: Country of Origin. We sort the bars by Region_Count descending.
            x=alt.X('Country of Origin:N', 
                    sort='-y', 
                    title='Country of Origin',
                    axis=alt.Axis(labelAngle=-55) # Replicates the rotation from the matplotlib version
                   ),
            # Y-axis: Region Count
            y=alt.Y('Region_Count:Q', title='Number of Unique Regions'),
            # Tooltip for interactivity
            tooltip=['Country of Origin', alt.Tooltip('Region_Count', title='Number of Unique Regions')],
            # Use color to visually separate the bars
            color=alt.Color('Country of Origin', legend=None) 
        ).properties(
            title='Count by Country of Origin'
        ).interactive(
            # Increase the width for better label readability, similar to the original figsize
            bind_y=False # Disable y-axis zoom/pan
        ).properties(width=800) 

        # Save the chart
        if filename:
            chart.save(filename)

        return chart
    create_country_region_altair_chart(df_country_region).save(fp="country_dist.png", scale_factor=2)
    return


@app.cell
def _(df_country_region):
    df_country_region
    return


@app.cell
def _(PCA, alt, np, pd, preprocessed_df):
    def plot_scree_plot(df: pd.DataFrame) -> alt.Chart:
        # 1. Fit PCA to get the explained variance for all possible components
        # The number of components is automatically the number of features.
        pca_model = PCA() 
        pca_model.fit(df)
    
        # Get the individual explained variance ratio
        explained_variance_ratio = pca_model.explained_variance_ratio_
    
        # 2. Create a DataFrame for Altair
        data = pd.DataFrame(
            {
                "Principal Component": np.arange(1, len(explained_variance_ratio) + 1),
                "Explained Variance Ratio": explained_variance_ratio,
            }
        )
    
        # 3. Create the Altair chart (Scree Plot)
        # Base chart setup
        base = alt.Chart(data).encode(
            # X-axis: Principal Component number
            x=alt.X(
                "Principal Component:O",  # Use 'O' for Ordinal to treat components as discrete categories
                axis=alt.Axis(title="Principal Component Number"),
            ),
            # Y-axis: Explained Variance Ratio
            y=alt.Y(
                "Explained Variance Ratio:Q",
                axis=alt.Axis(format=".1%", title="Individual Explained Variance"),
            ),
            tooltip=[
                "Principal Component",
                alt.Tooltip("Explained Variance Ratio", format=".4f"),
            ],
        ).properties(width=1000)

        # Line marks (to show the trend)
        line = base.mark_line(point=True).encode(
            color=alt.value("darkorange")
        )
    
        # Add text labels above the points for clarity
        text = base.mark_text(
            align='left',
            baseline='middle',
            dx=5, # Nudge text to the right
            dy=-10 # Nudge text up
        ).encode(
            text=alt.Text("Explained Variance Ratio", format=".1%"),
            color=alt.value("black")
        )

        # 4. Combine line and text and make it interactive
        final_chart = (line).properties(
            title="Scree Plot"
        ).interactive(bind_y=False) # Allow x-axis zoom/pan

        return final_chart

    plot_scree_plot(preprocessed_df).save(fp="scree_plot.png", scale_factor=2)
    return


@app.cell
def _(preprocessed_df):
    preprocessed_df
    return


@app.cell
def _(PCA, alt, np, pd, preprocessed_df):
    def plot_PCA_components(df: pd.DataFrame, n: int) -> alt.Chart:
        """df: Preprocessed dataframe
        Generates a plot to choose the appropriate number of components to pass to PCA
        """
        pca_model = PCA(n)
        pca_model.fit_transform(df)
        cum_sum_ratio = pca_model.explained_variance_ratio_.cumsum()
        # Create a DataFrame
        data = pd.DataFrame(
            {
                "Number of Components": np.arange(1, len(cum_sum_ratio) + 1),
                "Cumulative Explained Variance": cum_sum_ratio,
            }
        )

        # Find the first component count that explains >= 90% of the variance
        threshold_90 = data[data["Cumulative Explained Variance"] >= 0.1].iloc[0]
        component_90 = int(threshold_90["Number of Components"])
        print(component_90)
        # Create the base chart
        chart = (
            alt.Chart(data)
            .encode(
                x=alt.X(
                    "Number of Components:Q",
                    axis=alt.Axis(title="Principal Components"),
                ),
                y=alt.Y(
                    "Cumulative Explained Variance:Q",
                    axis=alt.Axis(
                        format=".0%",
                        title="Cumulative Explained Variance",
                    ),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    "Number of Components",
                    alt.Tooltip("Cumulative Explained Variance", format=".2%"),
                ],
            )
            .properties(title="Principal Components")
        )

        # Add the line
        line = chart.mark_line(point=True).encode(color=alt.value("darkblue"))

        rule_90_y = (
            alt.Chart(pd.DataFrame({"y": [0.9]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(y="y")
        )

        rule_90_x = (
            alt.Chart(pd.DataFrame({"x": [component_90]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(x=alt.X("x:Q"))
        )

        # Combine all layers
        final_chart = (line).interactive()

        return final_chart

    plot_PCA_components(preprocessed_df, 12)
    
    # .save(fp="cum_variance.png", scale_factor=2)
    return


@app.cell
def _(
    CountEncoder,
    OneHotEncoder,
    PCA,
    StandardScaler,
    df,
    fixup_data,
    make_column_transformer,
    make_pipeline,
    pd,
):
    fixed_df = fixup_data(df).drop(columns=["ID"])
    numerical_pipeline = make_pipeline(StandardScaler())
    frequency_encoding_pipeline = make_pipeline(CountEncoder())
    ohe_pipeline = make_pipeline(OneHotEncoder(sparse_output=False))
    numerical_col = fixed_df.select_dtypes(include=["float64"]).columns

    pre_processor = make_column_transformer(
        (numerical_pipeline, numerical_col),
        (frequency_encoding_pipeline, ["Region"]),
        (ohe_pipeline, ["Country of Origin", "Color"]),
        remainder="passthrough"
    )

    preprocessed_df = pd.DataFrame(
        data=pre_processor.fit_transform(fixed_df),
        columns=pre_processor.get_feature_names_out(),
        index=fixed_df.index
    )

    pca = PCA(n_components=5)
    pca_features = pca.fit_transform(preprocessed_df)
    pca_df = pd.DataFrame(pca_features, index=fixed_df.index)
    return fixed_df, pca_df, preprocessed_df


@app.cell
def _(pd):
    def label_data(clustering, df: pd.DataFrame) -> pd.DataFrame:
        labeld_df = df.copy(deep=True)
        labeld_df["Cluster"] = clustering.labels_
        return labeld_df
    return (label_data,)


@app.cell
def _(AgglomerativeClustering, df, label_data, pca_df):
    label_data(AgglomerativeClustering().fit(pca_df), df)
    return


@app.cell
def _(DBSCAN, df, label_data, pca_df):
    label_data(DBSCAN(eps=1, min_samples=10).fit(pca_df), df)
    return


@app.cell
def _(MeanShift, df, label_data, pca_df):
    label_data(MeanShift().fit(pca_df), df)
    return


@app.cell
def _(AgglomerativeClustering, pca_df, preprocessed_df, silhouette_score):
    silhouette_score(pca_df, AgglomerativeClustering(n_clusters=2).fit_predict(preprocessed_df))
    return


@app.cell
def _(DBSCAN, pca_df, silhouette_score):
    silhouette_score(pca_df, DBSCAN(eps=6, min_samples=6).fit_predict(pca_df))
    return


@app.cell
def _(MeanShift, pca_df, preprocessed_df, silhouette_score):
    silhouette_score(pca_df, MeanShift().fit_predict(preprocessed_df))
    return


@app.cell
def _(
    AgglomerativeClustering,
    DBSCAN,
    MeanShift,
    pca_df,
    preprocessed_df,
    silhouette_score,
):
    print(silhouette_score(pca_df, AgglomerativeClustering().fit_predict(preprocessed_df)))
    print(silhouette_score(pca_df, DBSCAN().fit_predict(preprocessed_df)))
    print(silhouette_score(pca_df, MeanShift().fit_predict(preprocessed_df)))
    return


@app.cell
def _(MeanShift, alt, df, label_data, pca_df, pd):
    def create_cluster_scatter_plot(pca_df: pd.DataFrame, original_df: pd.DataFrame, clustering_model, clustering_name: str ) -> alt.Chart:
        """
        Plots the clustered data in a 2D scatter plot using the first two Principal Components (PC1 and PC2).

        Args:
            pca_df: DataFrame containing the Principal Component features.
            original_df: The original, pre-fixup DataFrame (df) used for original features.
            clustering_model: A fitted clustering model (e.g., MeanShift().fit(pca_df)).
            filename: Optional filename to save the chart as.

        Returns:
            An Altair Chart object.
        """
        # 1. Get the clustered data and merge the PCA components with the original features
        clustered_data = label_data(clustering_model, original_df)
    
        # Rename PCA columns for clarity and merge. Assuming pca_df columns are not yet named.
        # The original pca_features array (fit_transform result) will have columns corresponding to PC1, PC2, etc.
        # Since pca_df was created from pca_features:
        pca_df.columns = [f"PC{i+1}" for i in range(pca_df.shape[1])]
    
        # Reset index of the clustered_data to ensure clean merge/join with pca_df index
        # We use join here as pca_df was created with index=fixed_df.index
        plot_df = clustered_data.join(pca_df)

        # 2. Define the list of tooltips
        tooltip_list = [
            'Country of Origin', 
            'Altitude',
            'Aroma', 
            'Flavor', 
            'Aftertaste', 
            'Acidity', 
            'Body', 
            'Balance', 
            'Overall',
            'Cluster:N'
        ]

        # 3. Create the Altair chart
        chart = alt.Chart(plot_df).mark_circle(size=60).encode(
            # X-axis: Principal Component 1
            x=alt.X('PC1:Q', title='Principal Component 1'),
        
            # Y-axis: Principal Component 2
            y=alt.Y('PC2:Q', title='Principal Component 2'),
        
            # Color points by Cluster. Treat Cluster as a Nominal (N) category.
            color=alt.Color('Cluster:N', title='Cluster'),
        
            # Tooltip for all required attributes
            tooltip=tooltip_list
        ).properties(
            title=f' {clustering_name} Clusters'
        ).interactive() # Enable zooming and panning


        return chart

    create_cluster_scatter_plot(pca_df, df, MeanShift().fit(pca_df), "Mean Shift")
    return (create_cluster_scatter_plot,)


@app.cell
def _(
    AgglomerativeCelustering,
    AgglomerativeClustering,
    DBSCAN,
    MeanShift,
    pca_df,
    preprocessed_df,
):
    mean_shift_model = MeanShift().fit(preprocessed_df)
    pca_mean_shift_model = MeanShift().fit(pca_df)
    agglomerative_model = AgglomerativeCelustering().fit(preprocessed_df) 
    pca_agglomertive_model = AgglomerativeClustering().fit(pca_df) 
    dbscan_model = DBSCAN(eps=2, min_samples=10).fit(preprocessed_df)
    pca_dbscan_model = DBSCAN(eps=2, min_samples=10).fit(pca_df)
    return


@app.cell
def _(AgglomerativeClustering, create_cluster_scatter_plot, df, pca_df):
    create_cluster_scatter_plot(pca_df, df, AgglomerativeClustering(n_clusters=3).fit(pca_df), "Agglomerative")
    return


@app.cell
def _(DBSCAN, create_cluster_scatter_plot, df, pca_df):
    create_cluster_scatter_plot(pca_df, df, DBSCAN(eps=2, min_samples=5).fit(pca_df), "DBSCAN")
    
    # .save(fp="DBSCAN_clusters.png", scale_factor=2)
    return


@app.cell
def _(alt, pca_df, pd):
    def create_pca_scatter_plot(pca_df: pd.DataFrame) -> alt.Chart:
        """
        Plots the data in a 2D scatter plot using the first two Principal Components (PC1 and PC2)
        without cluster coloring.

        Args:
            pca_df: DataFrame containing the Principal Component features.
    
        Returns:
            An Altair Chart object.
        """
        # Rename PCA columns for clarity. Assuming pca_df was created from pca_features:
        pca_df.columns = [f"PC{i + 1}" for i in range(pca_df.shape[1])]

        # 1. Create the base Altair chart
        chart = alt.Chart(pca_df).mark_circle(size=60).encode(
            # X-axis: Principal Component 1
            x=alt.X('PC1:Q', title='Principal Component 1'),
            # Y-axis: Principal Component 2
            y=alt.Y('PC2:Q', title='Principal Component 2'),
            # Tooltip for all required attributes (only PC1 and PC2 are available here)
            tooltip=['PC1', 'PC2']
        ).properties(
            title='Data Scatter Plot'
        ).interactive() # Enable zooming and panning

        return chart

    create_pca_scatter_plot(pca_df).save(fp="non_cluster_scatter.png", scale_factor=2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
