import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from ucimlrepo import fetch_ucirepo
    import altair as alt
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    from scipy.stats import ttest_ind
    from scipy.stats import chi2, chi2_contingency
    from typing import List, Optional
    import numpy as np
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
    return (
        ConfusionMatrixDisplay,
        GradientBoostingClassifier,
        LogisticRegression,
        OrdinalEncoder,
        RandomForestClassifier,
        alt,
        chi2_contingency,
        confusion_matrix,
        f1_score,
        fetch_ucirepo,
        mo,
        np,
        pd,
        recall_score,
        train_test_split,
        ttest_ind,
    )


@app.cell
def _(mo):
    mo.md(r"""## EDA""")
    return


@app.cell
def _(decode, features):
    decoded_df = decode(features)
    decoded_df
    return


@app.cell
def _(alt, pd):
    def plot_drug_scale_distribution_altair(
        df: pd.DataFrame,
        title: str = "Proportional Distribution of Usage Levels (CL0 to CL6) Per Drug",
    ) -> alt.Chart:
        """
        Plots a stacked bar chart using Altair for string-based CL values.

        Args:
            df (pd.DataFrame): The DataFrame where ALL columns are drug/usage features
                               with string values ('CL0' to 'CL6').
            title (str): The title for the Altair chart.

        Returns:
            alt.Chart: The Altair chart object.
        """

        # --- 1. Data Preparation and Melt ---

        # Automatically retrieve all column names
        drug_columns = df.columns.tolist()

        # Create a copy and map the string CL values to their ordinal integers (0-6)
        df_transformed = df[drug_columns].copy()

        # Apply a function to convert 'CLX' string to the integer X
        def cl_to_int(cl_string):
            if isinstance(cl_string, str) and cl_string.startswith("CL"):
                return int(cl_string[2:])
            # Handle potential NaNs or other unexpected values if they exist
            return None

        # Apply the conversion across all relevant columns
        df_transformed = df_transformed.map(cl_to_int)

        # Melt the DataFrame:
        df_melted = df_transformed.melt(
            value_vars=drug_columns,
            var_name="Drug",
            value_name="Usage_Level",  # This is now the integer (0-6)
        ).dropna(subset=["Usage_Level"])  # Drop rows if the conversion failed

        # Ensure Usage_Level is treated as an ordinal integer
        df_melted["Usage_Level"] = df_melted["Usage_Level"].astype(int)

        # 2. Define the Usage Level Order and Labels for the legend

        usage_order = list(range(7))

        # 3. Create the Altair Stacked Bar Chart

        chart = (
            alt.Chart(df_melted)
            .mark_bar()
            .encode(
                # X-axis: Drug Name, with rotation
                x=alt.X(
                    "Drug:N", axis=alt.Axis(title="Drug Type", labelAngle=-45)
                ),
                # Y-axis: Count normalized to sum to 1 (proportions)
                y=alt.Y(
                    "count():Q",
                    stack="normalize",
                    axis=alt.Axis(title="Proportion of Population", format="%"),
                ),
                # Color: Usage Level (Ordinal)
                color=alt.Color(
                    "Usage_Level:O",
                    sort=usage_order,
                    legend=alt.Legend(
                        title="Usage Level (Ordinal)",
                        # Use an expression to display the CLX label and its meaning
                        labelExpr="datum.label == '0' ? 'CL0: Never Used' : "
                        + "datum.label == '1' ? 'CL1: > 10 Years Ago' : "
                        + "datum.label == '2' ? 'CL2: > 1 Year Ago' : "
                        + "datum.label == '3' ? 'CL3: > 1 Month Ago' : "
                        + "datum.label == '4' ? 'CL4: > 1 Week Ago' : "
                        + "datum.label == '5' ? 'CL5: > 1 Day Ago' : "
                        + "datum.label == '6' ? 'CL6: Used Yesterday' : datum.label",
                    ),
                    scale=alt.Scale(range="ordinal"),
                ),
            )
            .properties(title=title)
            .interactive()
        )

        return chart
    return (plot_drug_scale_distribution_altair,)


@app.cell
def _(alt, dataset, pd):
    def generate_correlation_matrix(
        df: pd.DataFrame, title: str = "Feature Correlation Matrix"
    ) -> alt.Chart:
        """
        Generates an Altair heatmap visualization of the correlation matrix for a pandas DataFrame.

        The function first calculates the correlation matrix (R) and then transforms it
        into a long format suitable for visualization. It uses Altair to create a heatmap
        with numerical labels for precise correlation values.

        Args:
            df: The input pandas DataFrame containing numerical features.
            title: The title for the generated chart.

        Returns:
            An Altair Chart object representing the correlation heatmap.
        """
        # 1. Calculate the correlation matrix, ensuring only numerical columns are used
        corr_matrix = df.corr(numeric_only=True).reset_index()

        # 2. Transform the wide-form correlation matrix to long-form for Altair
        corr_data = corr_matrix.melt(
            id_vars="index", var_name="variable2", value_name="correlation"
        ).rename(columns={"index": "variable1"})

        # 3. Create the base chart
        base = (
            alt.Chart(corr_data)
            .encode(
                # Encode variables on X and Y axes (nominal type)
                x=alt.X("variable1:N", title=None),
                y=alt.Y("variable2:N", title=None),
            )
            .properties(title=title)
        )

        # 4. Create the heatmap (rectangles colored by correlation)
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(range="diverging", domain=[-1, 1], type="linear"),
                legend=alt.Legend(title="Correlation (R)"),
            ),
            tooltip=[
                alt.Tooltip("variable1:N", title="Feature 1"),
                alt.Tooltip("variable2:N", title="Feature 2"),
                alt.Tooltip("correlation:Q", title="Correlation", format=".2f"),
            ],
        )

        # 5. Add text labels for the exact correlation value
        # Hide the text for correlations of 1.0 (self-correlation) for cleaner look
        text = base.mark_text().encode(
            text=alt.Text("correlation:Q", format=".2f"),
            color=alt.condition(
                # White text for dark squares, black text for light squares
                alt.datum.correlation > 0.5,
                alt.value("white"),
                alt.value("black"),
            ),
            # Filter out the diagonal where correlation is 1 (variable with itself)
            opacity=alt.condition(
                alt.datum.correlation < 1.0, alt.value(1), alt.value(0)
            ),
        )

        # 6. Combine the heatmap and text layers and return
        return (
            (heatmap + text)
            .configure_axis(
                labelAngle=-45,  # Tilt x-axis labels for readability
                domain=False,  # Hide the domain line
                tickBand="extent",  # Make the ticks align to the middle of the rects
            )
            .configure_title(fontSize=18, anchor="start")
        )


    generate_correlation_matrix(dataset["features"])
    return


@app.cell
def _(plot_drug_scale_distribution_altair, target):
    plot_drug_scale_distribution_altair(target).save(fp="what.png", scale_factor=2)
    return


@app.cell
def _(alt, pd):
    def visualize_spearman_correlation_altair_df(
        df_predictors: pd.DataFrame, df_targets: pd.DataFrame
    ) -> alt.Chart:
        """
        Calculates the Spearman Rank Correlation between predictor features (df_predictors)
        and ordinal target usage levels (df_targets) and visualizes the result as an
        Altair heatmap.

        Args:
            df_predictors (pd.DataFrame): DataFrame containing only the predictor features (X).
            df_targets (pd.DataFrame): DataFrame containing only the drug usage levels (Y, 0-6).

        Returns:
            alt.Chart: The Altair chart object.
        """

        # 1. Combine DataFrames for Correlation Calculation
        # Ensure indices align, which is critical when merging dataframes.
        if not df_predictors.index.equals(df_targets.index):
            raise ValueError(
                "Indices of df_predictors and df_targets must be identical."
            )

        df_combined = pd.concat([df_predictors, df_targets], axis=1)

        predictor_cols = df_predictors.columns.tolist()
        target_cols = df_targets.columns.tolist()

        # 2. Calculate Spearman Correlation Matrix

        # Calculate the full Spearman correlation matrix
        full_corr_matrix = df_combined.corr(method="spearman")

        # Extract the required section: Predictors (Rows) vs. Targets (Columns)
        correlation_subset = full_corr_matrix.loc[predictor_cols, target_cols]

        # 3. Transform Matrix to Long Format (required by Altair)

        corr_long = correlation_subset.reset_index().rename(
            columns={"index": "Predictor"}
        )
        corr_long = corr_long.melt(
            id_vars="Predictor",
            var_name="Drug",  # Renamed Target to 'Drug' for clearer axis label
            value_name="Correlation",
        )

        # 4. Create the Altair Heatmap

        base = alt.Chart(corr_long).encode(
            x=alt.X(
                "Drug:N",
                axis=alt.Axis(title="Drug Usage (Ordinal Target)", labelAngle=-45),
            ),
            y=alt.Y("Predictor:N", axis=alt.Axis(title="Predictor Feature")),
            tooltip=[
                "Predictor:N",
                "Drug:N",
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )

        # Layer 1: The Heatmap
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(domain=[-1, 0, 1], range="diverging"),
                legend=alt.Legend(title="Spearman ρ"),
            )
        )

        # Layer 2: Annotation Text
        text = base.mark_text(baseline="middle").encode(
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Correlation > 0.6, alt.value("white"), alt.value("black")
            ),
        )

        # Combine layers and add title
        chart = (
            (heatmap + text)
            .properties(
                title="Spearman Rank Correlation (ρ) Between Predictors and Drug Usage Levels"
            )
            .interactive()
        )

        return chart
    return (visualize_spearman_correlation_altair_df,)


@app.cell
def _(features, target_encoded, visualize_spearman_correlation_altair_df):
    visualize_spearman_correlation_altair_df(
        df_predictors=features, df_targets=target_encoded
    )
    return


@app.cell
def _(OrdinalEncoder, pd):
    def ordinal_encode_drug_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies scikit-learn's OrdinalEncoder to the drug usage columns (CL0 to CL6)
        to convert the string labels into ordinal integers (0 to 6).

        Args:
            df (pd.DataFrame): The DataFrame where ALL columns are the drug/usage
                               features with string values ('CL0' to 'CL6').

        Returns:
            pd.DataFrame: A new DataFrame with the same columns, but with integer
                          values (0-6).
        """

        # 1. Define the correct, explicit ordinal order for the categories
        # The order must be a list of lists, where the inner list is the category
        # order for one column. Since all 19 columns have the SAME order, we repeat it.

        single_category_order = [
            f"CL{i}" for i in range(7)
        ]  # ['CL0', 'CL1', ..., 'CL6']

        # Create the categories list for all columns: 19 times the single order list
        drug_columns = df.columns.tolist()
        all_categories = [single_category_order] * len(drug_columns)

        # 2. Instantiate and fit the OrdinalEncoder

        # categories=all_categories ensures CL0 -> 0, CL1 -> 1, etc., regardless of alphabetical order.
        encoder = OrdinalEncoder(categories=all_categories, dtype="int32")

        # 3. Apply the transformation

        # OrdinalEncoder requires a 2D array and returns a NumPy array.
        # We fit and transform the data and replace the columns in the DataFrame.

        encoded_values = encoder.fit_transform(df[drug_columns])

        # Create the new DataFrame from the encoded values, preserving column names
        encoded_df = pd.DataFrame(
            encoded_values, columns=drug_columns, index=df.index
        )

        return encoded_df
    return (ordinal_encode_drug_data,)


@app.cell
def _(fetch_ucirepo, train_test_split):
    dataset = fetch_ucirepo(id=373)["data"]
    features = dataset["features"]
    target = dataset["targets"]

    # Holdout
    X_train, X_test, y_train, y_test = train_test_split(
        features, target["alcohol"], test_size=0.70, random_state=42, shuffle=True
    )
    return X_test, X_train, dataset, features, target, y_test, y_train


@app.cell
def _(ordinal_encode_drug_data, target):
    target_encoded = ordinal_encode_drug_data(target)
    return (target_encoded,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2 sample t-test

    This test it used for a continuous features for the predictor datasets
    """
    )
    return


@app.cell
def _(X_test, X_train, validate_continuous_partition):
    validate_continuous_partition(
        X_train=X_train["education"], X_test=X_test["education"]
    )
    return


@app.cell
def _(validate_categorical_partition, y_test, y_train):
    validate_categorical_partition(y_train=y_train, y_test=y_test)
    return


@app.cell
def _(chi2_contingency, np, pd, ttest_ind):
    def validate_categorical_partition(y_train, y_test, alpha=0.05):
        test_type = "Chi-Squared (Categorical)"
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        train_counts = pd.Series(y_train).value_counts()
        test_counts = pd.Series(y_test).value_counts()

        contingency_table = np.array([train_counts.values, test_counts.values])

        # 2. Perform Chi-Squared test
        statistic, p_value, _, _ = chi2_contingency(contingency_table)

        # 3. Determine the conclusion
        if p_value < alpha:
            conclusion = f"FAIL: Reject H₀ (p < {alpha}). Proportions are STATISTICALLY DIFFERENT."
        else:
            conclusion = f"PASS: Fail to reject H₀ (p >= {alpha}). Proportions are STATISTICALLY SIMILAR."

        return (test_type, statistic, p_value, conclusion)


    def validate_continuous_partition(X_train, X_test, alpha=0.05):
        test_type = "Two-Sample T-Test (Continuous)"
        # Convert input (which can be Pandas Series) to NumPy arrays for ttest_ind
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

        statistic, p_value = ttest_ind(X_train, X_test)

        # 3. Determine the conclusion
        if p_value < alpha:
            conclusion = f"FAIL: Reject H₀ (p < {alpha}). Means are STATISTICALLY DIFFERENT."
        else:
            conclusion = f"PASS: Fail to reject H₀ (p >= {alpha}). Means are STATISTICALLY SIMILAR."

        return (test_type, statistic, p_value, conclusion)
    return validate_categorical_partition, validate_continuous_partition


@app.cell(hide_code=True)
def _(pd):
    def decode(df: pd.DataFrame) -> pd.DataFrame:
        # 1. Demographic Mappings
        age_map = {
            round(-0.95197, 3): "18-24",
            round(-0.07854, 3): "25-34",
            round(0.49788, 3): "35-44",
            round(1.09449, 3): "45-54",
            round(1.82213, 3): "55-64",
            round(2.59171, 3): "65+",
        }

        gender_map = {round(0.48246, 3): "Female", round(-0.48246, 3): "Male"}

        education_map = {
            round(-2.43591, 3): "Left school before 16 years",
            round(-1.73790, 3): "Left school at 16 years",
            round(-1.43719, 3): "Left school at 17 years",
            round(-1.22751, 3): "Left school at 18 years",
            round(
                -0.61113, 3
            ): "Some college or university, no certificate or degree",
            round(-0.05921, 3): "Professional certificate/ diploma",
            round(0.45468, 3): "University degree",
            round(1.16365, 3): "Masters degree",
            round(1.98437, 3): "Doctorate degree",
        }

        country_map = {
            round(-0.09765, 3): "Australia",
            round(0.24923, 3): "Canada",
            round(-0.46841, 3): "New Zealand",
            round(-0.28519, 3): "Other",
            round(0.21128, 3): "Republic of Ireland",
            round(0.96082, 3): "UK",
            round(-0.57009, 3): "USA",
        }

        ethnicity_map = {
            round(-0.50212, 3): "Asian",
            round(-1.10702, 3): "Black",
            round(1.90725, 3): "Mixed-Black/Asian",
            round(0.12600, 3): "Mixed-White/Asian",
            round(-0.22166, 3): "Mixed-White/Black",
            round(0.11440, 3): "Other",
            round(-0.31685, 3): "White",
        }

        # 2. Psychometric Score Mappings (using integer scores as the target)
        nscore_map = {
            -3.464: 12,
            -3.157: 13,
            -2.757: 14,
            -2.522: 15,
            -2.423: 16,
            -2.344: 17,
            -2.218: 18,
            -2.050: 19,
            -1.870: 20,
            -1.692: 21,
            -1.551: 22,
            -1.439: 23,
            -1.328: 24,
            -1.194: 25,
            -1.053: 26,
            -0.921: 27,
            -0.792: 28,
            -0.678: 29,
            -0.580: 30,
            -0.467: 31,
            -0.348: 32,
            -0.246: 33,
            -0.149: 34,
            -0.052: 35,
            0.043: 36,
            0.136: 37,
            0.224: 38,
            0.313: 39,
            0.417: 40,
            0.521: 41,
            0.630: 42,
            0.735: 43,
            0.826: 44,
            0.911: 45,
            1.021: 46,
            1.133: 47,
            1.235: 48,
            1.373: 49,
            1.492: 50,
            1.604: 51,
            1.720: 52,
            1.840: 53,
            1.984: 54,
            2.127: 55,
            2.286: 56,
            2.463: 57,
            2.611: 58,
            2.822: 59,
            3.274: 60,
        }

        impulsive_map = {
            -2.555: "Low Impulsivity",
            -1.380: "Low-Medium",
            -0.711: "Medium-Low",
            -0.217: "Medium",
            0.193: "Medium-High",
            0.530: "High-Medium",
            0.881: "High",
            1.292: "Very High",
            1.862: "Extremely High",
            2.902: "Max Impulsivity",
        }

        ss_map = {
            -2.078: "Low SS",
            -1.549: "Low-Medium SS",
            -1.181: "Medium-Low SS",
            -0.846: "Medium SS",
            -0.526: "Medium-High SS",
            -0.216: "High-Medium SS",
            0.080: "High SS",
            0.401: "Very High SS",
            0.765: "Extremely High SS",
            1.225: "Max SS (Tier 1)",
            1.922: "Max SS (Tier 2)",
        }

        escore_map = {
            -3.274: 16,
            -3.005: 18,
            -2.728: 19,
            -2.538: 20,
            -2.449: 21,
            -2.323: 22,
            -2.211: 23,
            -2.114: 24,
            -2.040: 25,
            -1.922: 26,
            -1.763: 27,
            -1.633: 28,
            -1.508: 29,
            -1.376: 30,
            -1.232: 31,
            -1.092: 32,
            -0.948: 33,
            -0.806: 34,
            -0.695: 35,
            -0.575: 36,
            -0.440: 37,
            -0.300: 38,
            -0.155: 39,
            0.003: 40,
            0.168: 41,
            0.322: 42,
            0.476: 43,
            0.638: 44,
            0.805: 45,
            0.962: 46,
            1.114: 47,
            1.286: 48,
            1.454: 49,
            1.585: 50,
            1.741: 51,
            1.939: 52,
            2.127: 53,
            2.323: 54,
            2.573: 55,
            2.860: 56,
            3.005: 58,
            3.274: 59,
        }

        oscore_map = {
            -3.274: 24,
            -2.860: 26,
            -2.632: 28,
            -2.399: 29,
            -2.211: 30,
            -2.090: 31,
            -1.975: 32,
            -1.829: 33,
            -1.681: 34,
            -1.555: 35,
            -1.424: 36,
            -1.276: 37,
            -1.119: 38,
            -0.976: 39,
            -0.847: 40,
            -0.717: 41,
            -0.583: 42,
            -0.452: 43,
            -0.318: 44,
            -0.178: 45,
            -0.019: 46,
            0.141: 47,
            0.293: 48,
            0.446: 49,
            0.583: 50,
            0.723: 51,
            0.883: 52,
            1.062: 53,
            1.240: 54,
            1.435: 55,
            1.657: 56,
            1.885: 57,
            2.153: 58,
            2.449: 59,
            2.902: 60,
        }

        ascore_map = {
            -3.464: 12,
            -3.157: 16,
            -3.005: 18,
            -2.902: 23,
            -2.788: 24,
            -2.702: 25,
            -2.538: 26,
            -2.354: 27,
            -2.218: 28,
            -2.078: 29,
            -1.926: 30,
            -1.772: 31,
            -1.621: 32,
            -1.480: 33,
            -1.343: 34,
            -1.212: 35,
            -1.075: 36,
            -0.917: 37,
            -0.761: 38,
            -0.606: 39,
            -0.453: 40,
            -0.302: 41,
            -0.155: 42,
            -0.017: 43,
            0.131: 44,
            0.288: 45,
            0.439: 46,
            0.590: 47,
            0.761: 48,
            0.942: 49,
            1.114: 50,
            1.286: 51,
            1.450: 52,
            1.611: 53,
            1.819: 54,
            2.040: 55,
            2.234: 56,
            2.463: 57,
            2.757: 58,
            3.157: 59,
            3.464: 60,
        }

        # --- APPLYING THE MAPPINGS ---

        # List of (source_column_name, target_column_name, map_dictionary)
        transformations = [
            ("age", age_map),
            ("gender", gender_map),
            ("education", education_map),
            ("country", country_map),
            ("ethnicity", ethnicity_map),
            ("nscore", nscore_map),
            ("escore", escore_map),
            ("oscore", oscore_map),
            ("ascore", ascore_map),
            ("impuslive", impulsive_map),
            ("ss", ss_map),
        ]

        # Create a copy of the DataFrame to avoid modifying the original
        decoded_df = df.copy()

        for col, mapping in transformations:
            decoded_df[col] = decoded_df[col].round(3).replace(mapping)

        return decoded_df
    return (decode,)


@app.cell
def _(features):
    features
    return


@app.cell
def _(mo):
    mo.md(r"""## Model Training""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Logistic Regression""")
    return


@app.cell
def _(LogisticRegression):
    def logistic_regression(X_train, y_train, X_test):
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        return predictions, model
    return (logistic_regression,)


@app.cell
def _(X_test, X_train, features, log_coef, logistic_regression, y_train):
    log_coef(clf=logistic_regression(X_train, y_train, X_test)[1], df=features)
    return


@app.cell
def _(mo):
    mo.md(r"""### Random Forest""")
    return


@app.cell
def _(RandomForestClassifier):
    def random_forest(X_train, y_train, X_test):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        return predictions, model
    return (random_forest,)


@app.cell
def _(X_test, X_train, features, pd, random_forest, y_train):
    pd.DataFrame(
        {
            "feature": features.columns,
            "importance": random_forest(X_train, y_train, X_test)[
                1
            ].feature_importances_,
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Gradient Boost""")
    return


@app.cell
def _(GradientBoostingClassifier):
    def gradient_boost(X_train, y_train, X_test):
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        return predictions, model
    return (gradient_boost,)


@app.cell
def _(X_test, X_train, features, gradient_boost, pd, y_train):
    pd.DataFrame(
        {
            "feature": features.columns,
            "importance": gradient_boost(X_train, y_train, X_test)[
                1
            ].feature_importances_,
        }
    )
    return


@app.cell
def _(X_test, X_train, logistic_regression, recall_score, y_test, y_train):
    recall_score(
        y_test,
        logistic_regression(X_train, y_train, X_test)[0],
        average="weighted",
    )
    return


@app.cell
def _(X_test, X_train, f1_score, logistic_regression, y_test, y_train):
    f1_score(
        y_test,
        logistic_regression(X_train, y_train, X_test)[0],
        average="weighted",
    )
    return


@app.cell
def _(X_test, X_train, random_forest, recall_score, y_test, y_train):
    recall_score(
        y_test, random_forest(X_train, y_train, X_test)[0], average="weighted"
    )
    return


@app.cell
def _(X_test, X_train, f1_score, random_forest, y_test, y_train):
    f1_score(
        y_test, random_forest(X_train, y_train, X_test)[0], average="weighted"
    )
    return


@app.cell
def _(X_test, X_train, gradient_boost, recall_score, y_test, y_train):
    recall_score(
        y_test, gradient_boost(X_train, y_train, X_test)[0], average="weighted"
    )
    return


@app.cell
def _(X_test, X_train, f1_score, gradient_boost, y_test, y_train):
    f1_score(
        y_test, gradient_boost(X_train, y_train, X_test)[0], average="weighted"
    )
    return


@app.cell
def _(np, pd):
    def log_coef(df, clf):
        coef_matrix = clf.coef_
        classes = clf.classes_
        cols = df.columns
        mean = np.mean(np.abs(coef_matrix), axis=0)

        # coef_df = pd.DataFrame(coef_matrix, index=classes, columns=cols)

        return pd.DataFrame({"feature": cols, "importance": mean})
    return (log_coef,)


@app.cell
def _(alt, pd):
    def log_coef_heatmap(coef_df: pd.DataFrame):
        coef_df = coef_df.rename_axis("Class")
        coef_df = coef_df.reset_index()

        coef_long = coef_df.melt(
            id_vars="Class", var_name="Feature", value_name="Coefficient"
        )
        base = (
            alt.Chart(coef_long)
            .encode(
                x=alt.X("Feature", sort=None),
                y=alt.Y("Class"),
                tooltip=[
                    "Class",
                    "Feature",
                    alt.Tooltip("Coefficient", format=".2f"),
                ],
            )
            .properties(title="Multi-Class Logistic Regression Coefficients")
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "Coefficient",
                scale=alt.Scale(range="diverging", domainMid=0),
                legend=alt.Legend(title="Coefficient Value"),
            )
        )
        text = base.mark_text().encode(
            text=alt.Text("Coefficient", format=".2f"),
            color=alt.value("black"),
        )

        # Combine the heatmap and text layers, and configure the axis angle
        return (heatmap + text).configure_axis(
            # Tilt x-axis labels for better readability
            labelAngle=-45
        )
    return


@app.cell
def _(alt, pd):
    def plot_feature_importance(df: pd.DataFrame, algo: str) -> alt.Chart:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y("importance:Q", title="Importance", sort="ascending"),
                x=alt.X("feature:N", title="Feature", sort=None),
                tooltip=["feature:N", "importance:Q"],
            )
            .properties(title=f"Feature Importance ({algo})", width=600, height=400)
            .interactive()
        )

        return chart.configure_axis(labelAngle=-45)
    return (plot_feature_importance,)


@app.cell
def _(
    X_test,
    X_train,
    features,
    log_coef,
    logistic_regression,
    plot_feature_importance,
    y_train,
):
    plot_feature_importance(
        log_coef(clf=logistic_regression(X_train, y_train, X_test)[1], df=features),
        algo="Logistic Regression"
    ).save(fp="log_res_features.png", scale_factor=2)
    return


@app.cell
def _(
    X_test,
    X_train,
    features,
    gradient_boost,
    pd,
    plot_feature_importance,
    y_train,
):
    plot_feature_importance(
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance": gradient_boost(X_train, y_train, X_test)[
                    1
                ].feature_importances_,
            }
        ),
        algo="Gradient Boost"
    ).save(fp="gradboost_features.png", scale_factor=2)
    return


@app.cell
def _(
    X_test,
    X_train,
    features,
    pd,
    plot_feature_importance,
    random_forest,
    y_train,
):
    plot_feature_importance(
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance": random_forest(X_train, y_train, X_test)[
                    1
                ].feature_importances_,
            }
        ),
        algo="Random Forest"
    ).save(fp="random_forest_features.png", scale_factor=2)
    return


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix, np):
    def generate_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        accuracy = np.trace(cm) / np.sum(cm)
        return accuracy, ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    return (generate_confusion_matrix,)


@app.cell
def _(X_test, X_train, random_forest, y_test, y_train):
    random_forest(X_train, y_train, X_test)[1].score(X_test, y_test)
    return


@app.cell
def _(X_test, X_train, gradient_boost, y_test, y_train):
    gradient_boost(X_train, y_train, X_test)[1].score(X_test, y_test)
    return


@app.cell
def _(X_test, X_train, logistic_regression, y_test, y_train):
    logistic_regression(X_train, y_train, X_test)[1].score(X_test, y_test)
    return


@app.cell
def _(
    X_test,
    X_train,
    generate_confusion_matrix,
    logistic_regression,
    y_test,
    y_train,
):
    import matplotlib.pyplot as plt
    generate_confusion_matrix(y_true=y_test, y_pred=logistic_regression(X_train, y_train, X_test)[0])[1].plot()
    # plt.show()
    plt.savefig("confusion_matrix_log_res.png")
    return (plt,)


@app.cell
def _(
    X_test,
    X_train,
    generate_confusion_matrix,
    plt,
    random_forest,
    y_test,
    y_train,
):
    generate_confusion_matrix(y_true=y_test, y_pred=random_forest(X_train, y_train, X_test)[0])[1].plot()
    plt.savefig("confusion_matrix_random_forest.png")
    return


@app.cell
def _(
    X_test,
    X_train,
    generate_confusion_matrix,
    gradient_boost,
    plt,
    y_test,
    y_train,
):
    generate_confusion_matrix(y_true=y_test, y_pred=gradient_boost(X_train, y_train, X_test)[0])[1].plot()
    plt.savefig("confusion_matrix_grad_boost.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
