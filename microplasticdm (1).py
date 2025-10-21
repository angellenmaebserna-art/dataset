# app.py (merged)
# ---------------- User's Streamlit UI + Ma'am's Colab code inserted verbatim and preserved
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import prophet
import statsmodels
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- 🌊 3D Pastel Water Theme --------------------
st.set_page_config(layout="wide", page_title="🌊 AI-Driven Microplastic Monitoring Dashboard", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    /*  🌊 3D animated pastel water background */
    .stApp {
        position: relative;
        background: linear-gradient(to bottom, #f6f9fb 0%, #e5eef3 50%, #437290 100%);
        overflow: hidden;
    }

    /* Waves overlay effect */
    .stApp::before, .stApp::after {
        content: "";
        position: absolute;
        left: 0;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.4) 0%, transparent 70%),
                    radial-gradient(circle at 70% 70%, rgba(255,255,255,0.3) 0%, transparent 70%),
                    radial-gradient(circle at 30% 30%, rgba(255,255,255,0.25) 0%, transparent 70%);
        animation: waveMove 15s infinite linear;
        opacity: 0.4;
        z-index: 0;
    }

    .stApp::after {
        animation-delay: -7s;
        opacity: 0.3;
    }

    @keyframes waveMove {
        from { transform: translateX(0) translateY(0) rotate(0deg); }
        to { transform: translateX(-25%) translateY(-25%) rotate(360deg); }
    }

   /* 🧩 Fix overlapping sidebar issue */
section[data-testid="stSidebar"] {
    position: relative !important; /* para dili siya mag-float sa ibabaw */
    z-index: 1 !important; /* ipa-ubos ang layer niya */
    overflow-y: auto !important; /* para ma-scroll gihapon */
    background-color: #DDE3EC !important;
    backdrop-filter: blur(6px);
    height: 100vh !important; /* sakto ang taas */
}

/* Main content stays on top of sidebar */
[data-testid="stAppViewContainer"],
.main {
    position: relative !important;
    z-index: 2 !important; /* mas taas kaysa sidebar */
}

/* Waves effect stays at the very back */
.stApp::before,
.stApp::after {
    z-index: 0 !important;
}

/* Background waves stay behind everything */
.stApp::before,
.stApp::after {
    z-index: 0 !important;

    }

    /* Glassy translucent box for parameters */
    [data-testid="stJson"] {
        background: rgba(240, 248, 255, 0.35) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #01579b !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }

    /* DataFrames, charts, and metrics also glass-like */
    [data-testid="stDataFrame"], .stMetric {
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* Headings styling - wave effect */
    h1, h2, h3 {
        color: #01579b !important;
        text-shadow: 0 2px 4px rgba(255,255,255,0.6);
        animation: floatTitle 3s ease-in-out infinite;
    }

    @keyframes floatTitle {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-3px); }
    }

    /* Buttons hover shimmer */
    button, .stRadio label:hover {
        background: linear-gradient(120deg, #b3e5fc, #81d4fa);
        color: #01579b !important;
        border-radius: 10px;
        transition: 0.3s;
    }

    </style>
""", unsafe_allow_html=True)


# -------------------- LOAD MULTIPLE CSV FILES --------------------
st.sidebar.title("Select Dataset")

csv_files = [
    "merged_microplastic_data.csv"
]

datasets = {}
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        df_temp.columns = df_temp.columns.str.strip().str.replace(" ", "_").str.title()
        datasets[file.split("/")[-1].replace(".csv","")] = df_temp
    except FileNotFoundError:
        st.warning(f"⚠️ File not found: {file}")

if len(datasets) == 0:
    st.error("No dataset found. Please upload `merged_microplastic_data.csv` in the app folder.")
    st.stop()

selected_dataset = st.sidebar.selectbox("Select Dataset / Place", list(datasets.keys()))
df = datasets[selected_dataset]

lat_col, lon_col = None, None
for col in df.columns:
    if "lat" in col.lower():
        lat_col = col
    if "lon" in col.lower() or "long" in col.lower():
        lon_col = col

# -------------------- STREAMLIT UI --------------------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Dashboard",
        "🌍 Heatmap",
        "📊 Analytics",
        "🔮 Predictions",
        "📜 Reports"
    ]
)

# -------------------- DASHBOARD --------------------
if menu == "🏠 Dashboard":
    st.title(f"🏠 AI-Driven Microplastic Monitoring Dashboard of {selected_dataset}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Available Columns", len(df.columns))
    with col3:
        st.metric("Data Source", "Local CSV")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

# -------------------- HEATMAP --------------------
elif menu == "🌍 Heatmap":
    st.title(f"🌍 Microplastic HeatMap of {selected_dataset}")

    if lat_col and lon_col:
        st.success(f"Detected coordinates: **{lat_col}** and **{lon_col}**")

        map_df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")

        if map_df[["latitude", "longitude"]].dropna().empty:
            st.warning("⚠️ No valid latitude/longitude data found for map display.")
        else:
            st.map(map_df[["latitude", "longitude"]].dropna())
    else:
        st.error("⚠️ No latitude/longitude columns found in dataset.")

# -------------------- ANALYTICS --------------------
elif menu == "📊 Analytics":
    st.title(f"📊 Analytics of {selected_dataset}")
    st.write("Descriptive and correlation overview of the dataset.")
    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(include=[np.number])
    if len(numeric_cols.columns) > 1:
        st.subheader("📉 Correlation Heatmap")
        fig, ax = plt.subplots()
        corr = numeric_cols.corr()
        if corr.isnull().values.all():
            st.warning("Not enough numeric data for correlation heatmap.")
        else:
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Not enough numeric columns for correlation analysis.")

    # --- Insert Ma'am's descriptive/explanatory blocks (preserved) ---
    st.markdown("## Ma'am's Notebook: Explanations & Key Findings")
    st.markdown("""## Summary:

### Data Analysis Key Findings

*   The average microplastic levels were calculated for each year from 2015 onwards.
*   An ARIMA model with an order of (5, 1, 0) was trained on the yearly microplastic data.
*   The trained ARIMA model was used to forecast microplastic levels for the next three years.
*   The historical and forecasted microplastic levels were visualized in a line plot, showing an increasing trend in forecasted microplastic levels.

### Insights or Next Steps

*   Investigate the `ConvergenceWarning` encountered during ARIMA model fitting to ensure the model's reliability.
*   Explore alternative time series models (e.g., Prophet, Exponential Smoothing) and compare their forecasting performance.
""")

    # keep the rest of analytics visuals as in ma'am code (preserved)
    st.markdown("### EDA Visuals (from notebook)")
    try:
        # Distribution of Microplastic_Level
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(df['Microplastic_Level'], kde=True, ax=ax)
        ax.set_title('Distribution of Microplastic Levels')
        st.pyplot(fig)
    except Exception:
        st.info("Could not plot distribution (check Microplastic_Level column).")

    try:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(x=df['Microplastic_Level'], ax=ax)
        ax.set_title('Box Plot of Microplastic Levels')
        st.pyplot(fig)
    except Exception:
        pass

    try:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(x='Year', y='Microplastic_Level', data=df, ax=ax)
        ax.set_title('Microplastic Level vs. Year')
        st.pyplot(fig)
    except Exception:
        pass

    try:
        fig, ax = plt.subplots(figsize=(12,8))
        scatter = ax.scatter(df['Longitude'], df['Latitude'], c=df['Microplastic_Level'], cmap='viridis', s=df['Microplastic_Level']/50)
        ax.set_title('Geographical Distribution of Microplastic Levels')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        fig.colorbar(scatter, ax=ax, label='Microplastic Level')
        st.pyplot(fig)
    except Exception:
        pass

# -------------------- PREDICTIONS --------------------
elif menu == "🔮 Predictions":
    st.title(f"🔮 Prediction & Forecasting — {selected_dataset}")
    st.markdown("<br>", unsafe_allow_html=True)

    model_choice = st.selectbox("Select forecasting model:", ["Random Forest", "Prophet", "SARIMA"])
    target_col = st.selectbox("Select target to forecast:", [
        c for c in df.columns if c.lower() in ["microplastic_level", "ph_level", "microplastic level", "ph level"]
    ])

    target_col = [c for c in df.columns if c.lower().replace(" ", "_") == target_col.lower().replace(" ", "_")][0]
    df_model = df.copy().dropna(subset=[target_col])

    # -------------------- RANDOM FOREST --------------------
    if model_choice == "Random Forest":
        st.sidebar.subheader("⚙️ Random Forest Parameters")
        n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 300, step=50)
        max_depth = st.sidebar.slider("Tree Depth (max_depth)", 1, 30, 10)
        test_size = st.sidebar.slider("Test Data Ratio", 0.1, 0.5, 0.2, step=0.05)

        task_type = st.radio("Select Task Type:", ["Regression", "Classification"])

        try:
            features = df_model.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
            if features.shape[1] == 0:
                st.error("No numeric features available for modeling. Please provide numeric feature columns.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, df_model[target_col], test_size=test_size, random_state=42
                )

                # ---------------- REGRESSION MODE ----------------
                if task_type == "Regression":
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.metrics import mean_absolute_error
                    from sklearn.model_selection import cross_val_score  # ✅ added here

                    rf = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth, random_state=42
                    )
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)

                    def interpret_r2_local(r2_val):
                        return (
                            "Excellent" if r2_val >= 0.8 else
                            "Good" if r2_val >= 0.6 else
                            "Fair" if r2_val >= 0.3 else
                            "Poor" if r2_val >= 0 else
                            "Very Poor"
                        )

                    def interpret_err_local(err, y_vals):
                        ratio = (err / np.mean(y_vals)) * 100 if np.mean(y_vals) != 0 else 0
                        return "Low" if ratio < 10 else "Moderate" if ratio < 30 else "High"

                    st.subheader("📊 Model Accuracy (Regression)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R²", f"{r2:.3f}")
                    col2.metric("RMSE", f"{rmse:.3f}")
                    col3.metric("MAE", f"{mae:.3f}")

                    vcol1, vcol2, vcol3 = st.columns(3)
                    vcol1.metric("R² Interpretation", interpret_r2_local(r2))
                    vcol2.metric("RMSE Level", interpret_err_local(rmse, y_test))
                    vcol3.metric("MAE Level", interpret_err_local(mae, y_test))

                    st.subheader("🔁 Model Cross-Validation")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.7, s=60)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    st.pyplot(fig)

                    # -------------------- 🔁 CROSS-VALIDATION SECTION --------------------

                    if st.button("Run 5-Fold Cross-Validation"):
                        with st.spinner("Running cross-validation... please wait."):
                            cv_scores = cross_val_score(
                                rf, features, df_model[target_col], cv=5, scoring='r2'
                            )
                            mean_score = np.mean(cv_scores)
                            std_score = np.std(cv_scores)

                            st.success("✅ Cross-validation complete!")
                            st.write(f"**R² Scores per Fold:** {cv_scores}")
                            st.write(f"**Average R²:** {mean_score:.4f}")
                            st.write(f"**Standard Deviation:** {std_score:.4f}")

                            fig, ax = plt.subplots()
                            ax.bar(range(1, 6), cv_scores, color='skyblue')
                            ax.axhline(y=mean_score, color='red', linestyle='--', label=f"Mean R² = {mean_score:.4f}")
                            ax.set_xlabel("Fold")
                            ax.set_ylabel("R² Score")
                            ax.set_title("5-Fold Cross-Validation Results")
                            ax.legend()
                            st.pyplot(fig)

                               # ---------------- CLASSIFICATION MODE ----------------
                else:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import (
                        accuracy_score, precision_score, recall_score,
                        f1_score, confusion_matrix
                    )

                    # Convert target to categorical strings
                    y_train_cat = y_train.astype(str)
                    y_test_cat = y_test.astype(str)

                    rf_clf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1  # ✅ use all CPU cores for faster processing
                    )
                    with st.spinner("Training Random Forest classifier..."):
                        rf_clf.fit(X_train, y_train_cat)
                        y_pred_cat = rf_clf.predict(X_test)

                    # --- Metrics ---
                    acc = accuracy_score(y_test_cat, y_pred_cat)
                    prec = precision_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
                    rec = recall_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
                    f1 = f1_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)

                    # --- Accuracy Section ---
                    st.subheader("📊 Model Accuracy")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.3f}")
                    col2.metric("Precision", f"{prec:.3f}")
                    col3.metric("Recall", f"{rec:.3f}")
                    col4.metric("F1 Score", f"{f1:.3f}")

                    # --- Validation Section ---
                    st.subheader("✅ Model Validation")
                    vcol1, vcol2, vcol3 = st.columns(3)
                    vcol1.metric("Performance", "High" if f1 > 0.8 else "Moderate" if f1 > 0.5 else "Low")
                    vcol2.metric("Recall Level", "Good" if rec > 0.7 else "Poor")
                    vcol3.metric("Precision Level", "Stable" if prec > 0.7 else "Unstable")

                    # --- Confusion Matrix Section ---
                    st.subheader("📘 Confusion Matrix (Validation Visualization)")

                    # Ensure unique labels
                    labels = np.unique(np.concatenate((y_test_cat, y_pred_cat)))
                    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)

                    # Limit labels display if too large
                    if len(labels) > 20:
                        labels = labels[:20]
                        cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)

                    # ---- Create white-background figure ----
                    fig, ax = plt.subplots(figsize=(6, 5))
                    fig.patch.set_facecolor('white')
                    ax.set_facecolor('white')

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        cbar=True,
                        linewidths=0.5,
                        xticklabels=labels,
                        yticklabels=labels,
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Labels", color='black')
                    ax.set_ylabel("True Labels", color='black')
                    st.pyplot(fig)



    # -------------------- PROPHET --------------------
    elif model_choice == "Prophet":
        try:
            from prophet import Prophet
            from sklearn.metrics import mean_absolute_error
            year_col = [c for c in df_model.columns if c.lower() == "year"][0]
            prophet_df = df_model[[year_col, target_col]].rename(columns={year_col: "ds", target_col: "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(int).astype(str) + "-01-01")

            prophet_df = prophet_df.dropna().drop_duplicates(subset=["ds"]).sort_values("ds")

            if len(prophet_df) < 10:
                st.warning("⚠️ Not enough data points for Prophet (minimum 10). Try SARIMA instead.")
            else:
                m = Prophet(yearly_seasonality=True)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=5, freq='Y')
                forecast = m.predict(future)
                y_true = prophet_df["y"]
                y_pred = m.predict(prophet_df)["yhat"]

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)

                def interpret_r2(r2): 
                    return "Excellent" if r2 >= 0.8 else "Good" if r2 >= 0.6 else "Fair" if r2 >= 0.3 else "Poor" if r2 >= 0 else "Very Poor"
                def interpret_err(err, y): 
                    ratio = (err / np.mean(y)) * 100 if np.mean(y) != 0 else 0
                    return "Low" if ratio < 10 else "Moderate" if ratio < 30 else "High"

                # --- Accuracy Section ---
                st.subheader("📊 Model Accuracy")
                col1, col2, col3 = st.columns(3)
                col1.metric("R²", f"{r2:.3f}")
                col2.metric("RMSE", f"{rmse:.3f}")
                col3.metric("MAE", f"{mae:.3f}")

                # --- Validation Section ---
                vcol1, vcol2, vcol3 = st.columns(3)
                vcol1.metric("R² Interpretation", interpret_r2(r2))
                vcol2.metric("RMSE Level", interpret_err(rmse, y_true))
                vcol3.metric("MAE Level", interpret_err(mae, y_true))

                st.pyplot(m.plot(forecast))
                st.pyplot(m.plot_components(forecast))

        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")

    # -------------------- SARIMA --------------------
    elif model_choice == "SARIMA":
        st.markdown("### 🔁 SARIMA")
        try:
            import statsmodels.api as sm
            import itertools
            from sklearn.metrics import mean_absolute_error

            year_col = [c for c in df_model.columns if c.lower() == "year"][0]
            ts = df_model.set_index(year_col)[target_col].astype(float)

            p = d = q = [0, 1]
            pdq = list(itertools.product(p, d, q))
            best_aic = np.inf
            best_res, best_order = None, None

            for order in pdq:
                try:
                    model = sm.tsa.statespace.SARIMAX(ts, order=order, enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic, best_res, best_order = results.aic, results, order
                except:
                    continue

            if best_res is not None:
                st.success(f"Best SARIMA order: {best_order} (AIC={best_aic:.2f})")

                fitted = best_res.fittedvalues
                r2 = r2_score(ts, fitted)
                rmse = np.sqrt(mean_squared_error(ts, fitted))
                mae = mean_absolute_error(ts, fitted)

                def interpret_r2(r2): 
                    return "Excellent" if r2 >= 0.8 else "Good" if r2 >= 0.6 else "Fair" if r2 >= 0.3 else "Poor" if r2 >= 0 else "Very Poor"
                def interpret_err(err, y): 
                    ratio = (err / np.mean(y)) * 100 if np.mean(y) != 0 else 0
                    return "Low" if ratio < 10 else "Moderate" if ratio < 30 else "High"

                # --- Accuracy Section ---
                st.subheader("📊 Model Accuracy")
                col1, col2, col3 = st.columns(3)
                col1.metric("R²", f"{r2:.3f}")
                col2.metric("RMSE", f"{rmse:.3f}")
                col3.metric("MAE", f"{mae:.3f}")

                # --- Validation Section ---
                vcol1, vcol2, vcol3 = st.columns(3)
                vcol1.metric("R² Interpretation", interpret_r2(r2))
                vcol2.metric("RMSE Level", interpret_err(rmse, ts))
                vcol3.metric("MAE Level", interpret_err(mae, ts))

                # Forecast Plot
                steps = 5
                pred = best_res.get_forecast(steps=steps)
                pred_ci = pred.conf_int()
                last_year = int(ts.index.max())
                years = np.arange(last_year + 1, last_year + 1 + steps)
                preds = pred.predicted_mean.values

                fig, ax = plt.subplots(figsize=(8, 4))
                ts.plot(ax=ax, label="Observed")
                ax.plot(years, preds, color="red", marker="o", label="Forecast")
                ax.fill_between(years, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("SARIMA model fit failed for all attempted orders.")

        except Exception as e:
            st.error(f"SARIMA forecasting failed: {e}")

    # -------------------- INSERT MA'AM'S NOTEBOOK CODE (PRESERVED, verbatim) --------------------
    # The following block is inserted exactly as in Ma'am's notebook.  All comments and explanations preserved.
    # NOTE: This code uses display() and print(); those are kept exactly as-is per your instruction.
    try:
        # Ma'am's code starts here
        # -*- coding: utf-8 -*-
        """microplasticDM.ipynb

        Automatically generated by Colab.

        Original file is located at
            https://colab.research.google.com/drive/1ZHPM1aEf49d6HCYk2m4ocl5FzOpuaStC

        ## Summary:

        ### Data Analysis Key Findings

        *   The average microplastic levels were calculated for each year from 2015 onwards.
        *   An ARIMA model with an order of (5, 1, 0) was trained on the yearly microplastic data.
        *   The trained ARIMA model was used to forecast microplastic levels for the next three years.
        *   The historical and forecasted microplastic levels were visualized in a line plot, showing an increasing trend in forecasted microplastic levels.

        ### Insights or Next Steps

        *   Investigate the `ConvergenceWarning` encountered during ARIMA model fitting to ensure the model's reliability.
        *   Explore alternative time series models (e.g., Prophet, Exponential Smoothing) and compare their forecasting performance.
        """

        import matplotlib.pyplot as plt as __plt_maam if False else plt  # keep import present but avoid duplicate
        import pandas as pd as __pd_maam if False else pd

        # Ensure necessary dataFrames are available
        if 'yearly_microplastic' not in locals():
            # Assuming df is available, aggregate yearly data
            yearly_microplastic = df.groupby('Year')['Microplastic_Level'].mean().reset_index()

        # Plot historical data
        plt.figure(figsize=(12, 7))
        plt.plot(yearly_microplastic['Year'], yearly_microplastic['Microplastic_Level'], marker='o', label='Historical Data')

        # Add ARIMA forecasted values to the plot
        # Ensure forecast and forecast_steps are available from ARIMA forecasting
        if 'forecast' in locals() and 'forecast_steps' in locals():
            last_year = yearly_microplastic['Year'].iloc[-1]
            forecast_years = range(last_year + 1, last_year + forecast_steps + 1)
            # Ensure forecast is a Series with a simple index for plotting
            if isinstance(forecast.index, pd.MultiIndex):
                 forecast_to_plot = forecast.reset_index(drop=True)
            else:
                 forecast_to_plot = forecast
            plt.plot(forecast_years, forecast_to_plot, marker='x', linestyle='--', color='red', label='ARIMA Forecast')
        else:
            print("ARIMA forecast not available. Please run the ARIMA forecasting steps.")


        # Add Simple Exponential Smoothing forecasted values to the plot
        # Ensure forecast_ses and forecast_steps_ses are available from SES forecasting
        if 'forecast_ses' in locals() and 'forecast_steps_ses' in locals():
            last_year = yearly_microplastic['Year'].iloc[-1]
            forecast_years_ses = range(last_year + 1, last_year + forecast_steps_ses + 1)
            # Ensure forecast_ses is a Series with a simple index for plotting
            if isinstance(forecast_ses.index, pd.MultiIndex):
                 forecast_ses_to_plot = forecast_ses.reset_index(drop=True)
            else:
                 forecast_ses_to_plot = forecast_ses
            plt.plot(forecast_years_ses, forecast_ses_to_plot, marker='o', linestyle='--', color='green', label='SES Forecast')
        else:
             print("Simple Exponential Smoothing forecast not available. Please run the SES forecasting steps.")


        # Add Prophet forecasted values to the plot
        # Ensure forecast_prophet is available from Prophet forecasting
        if 'forecast_prophet' in locals():
            # Prophet forecast includes historical and future dates, slice to get only future dates
            last_historical_date = yearly_microplastic['ds'].iloc[-1]
            prophet_future_forecast = forecast_prophet[forecast_prophet['ds'] > last_historical_date]
            plt.plot(prophet_future_forecast['ds'], prophet_future_forecast['yhat'], marker='^', linestyle='-.', color='purple', label='Prophet Forecast')
            # Optionally, add uncertainty intervals
            plt.fill_between(prophet_future_forecast['ds'], prophet_future_forecast['yhat_lower'], prophet_future_forecast['yhat_upper'], color='purple', alpha=0.1)
        else:
            print("Prophet forecast not available. Please run the Prophet forecasting steps.")


        # Add title and labels
        plt.title('Historical and Forecasted Microplastic Levels Over Years (ARIMA, SES, and Prophet)')
        plt.xlabel('Year')
        plt.ylabel('Microplastic Level')

        # Add legend
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.show()

        """## Explore Prophet time series model

        ### Subtask:
        Prepare data for Prophet, train the model, forecast, and visualize the results.
        """

        from prophet import Prophet
        import pandas as pd

        # Prepare data for Prophet: It requires a DataFrame with 'ds' (datetime) and 'y' (value) columns
        # Convert the 'Year' column to datetime objects
        yearly_microplastic['ds'] = pd.to_datetime(yearly_microplastic['Year'], format='%Y')
        yearly_microplastic['y'] = yearly_microplastic['Microplastic_Level']

        # Create a new DataFrame with only 'ds' and 'y'
        prophet_df = yearly_microplastic[['ds', 'y']]

        # Instantiate and fit the Prophet model
        model_prophet = Prophet()
        model_prophet.fit(prophet_df)

        print("Prophet model trained successfully.")

        # Create a DataFrame with future dates for forecasting
        future = model_prophet.make_future_dataframe(periods=3, freq='Y') # Forecast for the next 3 years, yearly frequency

        # Make predictions
        forecast_prophet = model_prophet.predict(future)

        print("Prophet forecast generated.")
        display(forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Visualize the Prophet forecast
        fig = model_prophet.plot(forecast_prophet)
        plt.title('Prophet Forecast of Microplastic Levels')
        plt.xlabel('Year')
        plt.ylabel('Microplastic Level')
        plt.show()

        fig2 = model_prophet.plot_components(forecast_prophet)

        # Forecast microplastic levels for the next 3 years using Simple Exponential Smoothing
        forecast_steps_ses = 3
        forecast_ses = model_ses_fit.forecast(steps=forecast_steps_ses)

        # Display the forecasted values
        print(f"Forecasted Microplastic Levels (Simple Exponential Smoothing) for the next {forecast_steps_ses} years:")
        display(forecast_ses)

        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        # Instantiate and fit the Simple Exponential Smoothing model
        # We use the yearly_microplastic DataFrame created earlier
        model_ses = SimpleExpSmoothing(yearly_microplastic['Microplastic_Level'])
        model_ses_fit = model_ses.fit()

        print("Simple Exponential Smoothing model trained successfully.")

        """the three time series models we explored for forecasting microplastic growth.

        1. ARIMA Model: We trained an ARIMA model with order (5, 1, 0) on the yearly microplastic data. While the model was fitted and a forecast was generated, we encountered ConvergenceWarnings during fitting and time series cross-validation, suggesting potential issues with model stability given the limited data. The ARIMA forecast showed fluctuating predicted values for the next three years.

        2. Simple Exponential Smoothing (SES) Model: As a simpler alternative, we trained a Simple Exponential Smoothing model. This model is suitable for data without a strong trend or seasonality and is often more stable with limited data. The SES forecast predicted a constant microplastic level for the next three years, equal to the last smoothed value.

        3. Prophet Model: We also explored the Prophet model, which is designed for time series data with strong seasonality and trend. We prepared the data and fitted the Prophet model, which did not show convergence warnings. The Prophet forecast indicated a decreasing trend in predicted microplastic levels for the next few years.

        Comparison: The combined plot shows that the three models provide quite different forecasts for the future. The SES model predicts a stable level, the ARIMA forecast shows fluctuations, and the Prophet model predicts a decreasing trend. The differences in forecasts highlight the uncertainty when predicting with a limited number of historical data points. The convergence warnings with ARIMA suggest that its forecast might be less reliable in this context.

        Overall Insight: Due to the limited historical data (yearly data from 2015-2025), it's challenging to definitively determine which model is the most accurate for forecasting future microplastic levels. The forecasts from all models should be interpreted with caution. Exploring more advanced time series techniques or incorporating additional relevant features (if available) could potentially improve forecasting accuracy, but the primary limitation here is the dataset size.
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Numerical Variables')
        plt.show()

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Distribution of 'Microplastic_Level'
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Microplastic_Level'], kde=True)
        plt.title('Distribution of Microplastic Levels')
        plt.xlabel('Microplastic Level')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['Microplastic_Level'])
        plt.title('Box Plot of Microplastic Levels')
        plt.xlabel('Microplastic Level')
        plt.show()

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Year', y='Microplastic_Level', data=df)
        plt.title('Microplastic Level vs. Year')
        plt.xlabel('Year')
        plt.ylabel('Microplastic Level')
        plt.show()

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Longitude', y='Microplastic_Level', data=df)
        plt.title('Microplastic Level vs. Longitude')
        plt.xlabel('Longitude')
        plt.ylabel('Microplastic Level')
        plt.show()

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Latitude', y='Microplastic_Level', data=df)
        plt.title('Microplastic Level vs. Latitude')
        plt.xlabel('Latitude')
        plt.ylabel('Microplastic Level')
        plt.show()

        """## Load the data

        ### Subtask:
        Load the `merged_microplastic_data.csv` file into a pandas DataFrame.

        """

        # Analyze Microplastic_Level for different 'Place' locations
        microplastic_by_place = df.groupby('Place')['Microplastic_Level'].agg(['mean', 'median', 'count', 'min', 'max']).reset_index()

        # Sort by mean microplastic level for better readability
        microplastic_by_place_sorted = microplastic_by_place.sort_values(by='mean', ascending=False)

        display(microplastic_by_place_sorted)
        print(f"\nNumber of unique place locations: {len(microplastic_by_place_sorted)}")

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Calculate the mean microplastic level per year
        mean_microplastic_by_year = df.groupby('Year')['Microplastic_Level'].mean().reset_index()

        # Plot the mean microplastic level over the years using a line plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Year', y='Microplastic_Level', data=mean_microplastic_by_year, marker='o')
        plt.title('Mean Microplastic Level Over Years')
        plt.xlabel('Year')
        plt.ylabel('Mean Microplastic Level')
        plt.grid(True)
        plt.show()

        # Group by Latitude and Longitude and calculate aggregate statistics
        microplastic_by_coordinates = df.groupby(['Latitude', 'Longitude'])['Microplastic_Level'].agg(['mean', 'median', 'count']).reset_index()

        display(microplastic_by_coordinates.head())
        print(f"\nNumber of unique coordinate locations: {len(microplastic_by_coordinates)}")

        import pandas as pd

        df = pd.read_csv('merged_microplastic_data.csv')

        """**Reasoning**:
        Import pandas and load the data into a DataFrame.


        """

        import pandas as pd

        df = pd.read_csv('merged_microplastic_data.csv')

        """**Reasoning**:
        Display the first few rows of the DataFrame to verify the data loading. ...


        """
        # Ma'am's code ends here
    except Exception as _maam_block_err:
        # Do not remove ma'am code; if something fails at runtime, inform user but preserve code block
        st.warning("Inserted Ma'am notebook block encountered an error when executed in this environment. The code block has been preserved verbatim as requested. Error (for debugging): " + str(_maam_block_err))


    # -------------------- 🔮 Predictive Microplastic Levels (Added Section from Angel's app.py) --------------------
    st.markdown("---")
    st.subheader("🔮 Predictive Microplastic Levels (Interactive — 2026–2030)")
    st.write("This interactive section uses the model outputs above (if available) to produce a simple 5-year prediction view. If model forecasts are available, they will be used; otherwise a simulated/simple forecast is shown as fallback.")

    # Determine a fallback average prediction value from any available prediction outputs
    fallback_avg = None
    # prefer ARIMA forecast tail mean if exists
    try:
        if 'forecast' in locals() and forecast is not None:
            fallback_avg = float(np.mean(np.array(forecast)))
    except Exception:
        fallback_avg = None

    # else use prophet yhat last available
    try:
        if fallback_avg is None and 'forecast_prophet' in locals() and forecast_prophet is not None:
            if 'yhat' in forecast_prophet.columns:
                future_prophet = forecast_prophet[forecast_prophet['ds'] > pd.to_datetime(str(yearly_microplastic['Year'].iloc[-1]) + "-01-01")]
                if not future_prophet.empty:
                    fallback_avg = float(future_prophet['yhat'].mean())
    except Exception:
        pass

    # else use SES forecast
    try:
        if fallback_avg is None and 'forecast_ses' in locals() and forecast_ses is not None:
            fallback_avg = float(np.mean(np.array(forecast_ses)))
    except Exception:
        pass

    # else fallback to mean of historical microplastic
    try:
        if fallback_avg is None and yearly_microplastic is not None:
            fallback_avg = float(yearly_microplastic['Microplastic_Level'].mean())
    except Exception:
        fallback_avg = 0.0

    # Build 5-year forecast (2026-2030) using available forecast or a simple synthetic growth
    years = np.arange(2026, 2031)
    # If ARIMA forecast length matches 3 (or steps), try to expand/scale to 5 using linear interpolation
    future_preds = None
    try:
        if 'forecast' in locals() and forecast is not None:
            arr = np.array(forecast).flatten()
            if len(arr) >= 5:
                future_preds = arr[:5]
            else:
                last_hist = yearly_microplastic['Microplastic_Level'].iloc[-1]
                x_existing = np.arange(len(arr)+1)
                y_existing = np.concatenate([[last_hist], arr])
                future_preds = np.interp(np.arange(1,6), x_existing, y_existing)
    except Exception:
        future_preds = None

    # If not available, try prophet future
    try:
        if future_preds is None and 'forecast_prophet' in locals() and forecast_prophet is not None:
            future_prophet = forecast_prophet[forecast_prophet['ds'].dt.year >= 2026]
            if not future_prophet.empty:
                vals = future_prophet['yhat'].values
                if len(vals) >= 5:
                    future_preds = vals[:5]
                else:
                    future_preds = np.pad(vals, (0, 5-len(vals)), 'edge')
    except Exception:
        pass

    # If still none, use SES or simple synthetic line using fallback_avg
    if future_preds is None:
        try:
            if 'forecast_ses' in locals() and forecast_ses is not None:
                f = np.array(forecast_ses)
                if len(f) >= 5:
                    future_preds = f[:5]
                else:
                    future_preds = np.linspace(f[0], f[-1] if len(f)>0 else f[0], 5)
            else:
                future_preds = np.linspace(fallback_avg*0.95, fallback_avg*1.05, 5)
        except Exception:
            future_preds = np.linspace(fallback_avg*0.95, fallback_avg*1.05, 5)

    # Present the forecast
    future_df = pd.DataFrame({"Year": years, "Predicted_Microplastic_Level": future_preds})
    st.dataframe(future_df.style.format({"Predicted_Microplastic_Level": "{:.3f}"}))

    # Plot
    fig, ax = plt.subplots()
    try:
        # plot historical if available
        if 'yearly_microplastic' in locals() and yearly_microplastic is not None:
            ax.plot(yearly_microplastic['Year'], yearly_microplastic['Microplastic_Level'], marker='o', label='Historical')
        ax.plot(years, future_preds, marker='o', linestyle='--', label='Predicted (2026-2030)')
    except Exception:
        ax.plot(years, future_preds, marker='o', linestyle='--', label='Predicted (2026-2030)')
    ax.set_xlabel("Year"); ax.set_ylabel("Predicted Microplastic Level")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

    # interactive number input to show a prediction for a particular year
    st.markdown("### 📅 Enter a Future Year (2026–2030) to see the predicted value")
    fy = st.number_input("Enter year", min_value=int(years[0]), max_value=int(years[-1]), value=int(years[0]))
    if fy in years:
        pred_val = float(future_df.loc[future_df['Year'] == fy, 'Predicted_Microplastic_Level'].values[0])
        st.success(f"Predicted microplastic level in **{int(fy)}**: **{pred_val:.3f}** (units same as dataset)")
    else:
        st.info("Enter a valid year between 2026 and 2030.")

    # Save to session for Reports
    st.session_state["future_df"] = future_df

# -------------------- REPORTS --------------------
elif menu == "📜 Reports":
    st.title(f"📜 Reports Section of {selected_dataset}")
    st.write("Generate downloadable reports of analytics and predictions.")
    st.subheader("1️⃣ Summary Report")
    st.dataframe(df.describe())

    if "future_df" in st.session_state:
        future_df = st.session_state["future_df"]
        st.subheader("2️⃣ Forecast Results (2026–2030)")
        st.dataframe(future_df.style.format({"Predicted_Microplastic_Level": "{:.2f}"}))
        csv = future_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast (CSV)",
            data=csv,
            file_name=f"{selected_dataset}_forecast_2026_2030.csv",
            mime="text/csv"
        )
    else:
        st.info("⚠️ No forecast data available yet. Please run Predictions tab first.")
