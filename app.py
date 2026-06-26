import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# PAGE CONFIGURATION
# ==========================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main{
    background:#F5F7FA;
}

.title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:18px;
}

.header{
    background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    padding:25px;
    border-radius:15px;
    margin-bottom:20px;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.15);
}

.stButton>button{
    width:100%;
    background:#0066cc;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#004a99;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================

@st.cache_data
def load_data():
    df = pd.read_csv("house price data.csv")
    return df

df = load_data()

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():
    with open("model.pkl","rb") as file:
        model = pickle.load(file)

    with open("metrics.pkl","rb") as file:
        metrics = pickle.load(file)

    return model, metrics

model, metrics = load_model()

# ==========================
# HEADER
# ==========================

st.markdown("""
<div class="header">
<h1 class="title">🏠 House Price Prediction Dashboard</h1>
<p class="subtitle">
Machine Learning | Streamlit | Random Forest Regressor
</p>
</div>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Visualizations",
        "Prediction",
        "Batch Prediction",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
This application predicts house prices using a trained
Random Forest Regression model.

Developer:
Safa Maryam
"""
)
# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("📊 Dashboard")

    # -------------------------------
    # Remove unwanted columns
    # -------------------------------

    dashboard_df = df.copy()

    for col in ["date", "street"]:
        if col in dashboard_df.columns:
            dashboard_df.drop(col, axis=1, inplace=True)

    # -------------------------------
    # KPI Cards
    # -------------------------------

    avg_price = dashboard_df["price"].mean()
    max_price = dashboard_df["price"].max()
    min_price = dashboard_df["price"].min()
    total_houses = len(dashboard_df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🏠 Total Houses",
        f"{total_houses:,}"
    )

    c2.metric(
        "💰 Average Price",
        f"${avg_price:,.0f}"
    )

    c3.metric(
        "📈 Highest Price",
        f"${max_price:,.0f}"
    )

    c4.metric(
        "📉 Lowest Price",
        f"${min_price:,.0f}"
    )

    st.markdown("---")

    # -------------------------------
    # Dataset Preview
    # -------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        dashboard_df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # -------------------------------
    # Dataset Information
    # -------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("📌 Dataset Information")

        info = pd.DataFrame({
            "Property": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows"
            ],
            "Value": [
                dashboard_df.shape[0],
                dashboard_df.shape[1],
                dashboard_df.isnull().sum().sum(),
                dashboard_df.duplicated().sum()
            ]
        })

        st.table(info)

    with right:

        st.subheader("📈 Statistical Summary")

        st.dataframe(
            dashboard_df.describe(),
            use_container_width=True
        )

    st.markdown("---")

    # -------------------------------
    # Numerical Features
    # -------------------------------

    st.subheader("🔢 Numerical Features")

    numerical_cols = dashboard_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    st.write(numerical_cols)

    st.markdown("---")

    # -------------------------------
    # Categorical Features
    # -------------------------------

    st.subheader("🏙️ Categorical Features")

    categorical_cols = dashboard_df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    st.write(categorical_cols)

    st.markdown("---")

    # -------------------------------
    # Model Performance
    # -------------------------------

    st.subheader("🤖 Model Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "R² Score",
        f"{metrics['R2']:.3f}"
    )

    m2.metric(
        "MAE",
        f"{metrics['MAE']:,.0f}"
    )

    m3.metric(
        "RMSE",
        f"{metrics['RMSE']:,.0f}"
    )

    m4.metric(
        "MSE",
        f"{metrics['MSE']:,.0f}"
    )

    st.markdown("---")

    # -------------------------------
    # Price Range
    # -------------------------------

    st.subheader("💵 House Price Range")

    st.success(
        f"""
Lowest Price : **${min_price:,.0f}**

Highest Price : **${max_price:,.0f}**

Average Price : **${avg_price:,.0f}**
"""
    )
# =====================================================
# VISUALIZATIONS
# =====================================================

elif page == "Visualizations":

    st.title("📊 Data Visualizations")

    chart_df = df.copy()

    # Remove unnecessary columns if they exist
    for col in ["date", "street"]:
        if col in chart_df.columns:
            chart_df.drop(columns=col, inplace=True)

    # -------------------------------------------------
    # Price Distribution
    # -------------------------------------------------

    st.subheader("💰 House Price Distribution")

    fig = px.histogram(
        chart_df,
        x="price",
        nbins=40,
        color_discrete_sequence=["royalblue"],
        title="Distribution of House Prices"
    )

    fig.update_layout(template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Bedrooms vs Price
    # -------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🛏 Bedrooms vs Price")

        fig = px.scatter(
            chart_df,
            x="bedrooms",
            y="price",
            color="bedrooms",
            title="Bedrooms vs House Price"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("🛁 Bathrooms vs Price")

        fig = px.scatter(
            chart_df,
            x="bathrooms",
            y="price",
            color="bathrooms",
            title="Bathrooms vs House Price"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Living Area vs Price
    # -------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("🏡 Living Area vs Price")

        fig = px.scatter(
            chart_df,
            x="sqft_living",
            y="price",
            color="floors",
            title="Living Area vs Price"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        st.subheader("🏢 Floors vs Price")

        fig = px.box(
            chart_df,
            x="floors",
            y="price",
            color="floors",
            title="Floors vs House Price"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Top Cities
    # -------------------------------------------------

    if "city" in chart_df.columns:

        st.subheader("🏙 Top 10 Cities")

        city_count = (
            chart_df["city"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        city_count.columns = ["City", "Count"]

        fig = px.bar(
            city_count,
            x="City",
            y="Count",
            color="Count",
            title="Top 10 Cities"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Condition Distribution
    # -------------------------------------------------

    if "condition" in chart_df.columns:

        st.subheader("🏠 House Condition Distribution")

        fig = px.pie(
            chart_df,
            names="condition",
            title="Condition Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = chart_df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Viridis",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hoverongaps=False
        )
    )

    heatmap.update_layout(
        height=700,
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(heatmap, use_container_width=True)

    # -------------------------------------------------
    # Dataset Insights
    # -------------------------------------------------

    st.subheader("📈 Quick Insights")

    left, right = st.columns(2)

    with left:

        st.info(f"""
### Dataset Summary

- Total Houses : **{len(chart_df):,}**
- Total Features : **{chart_df.shape[1]}**
- Average Price : **${chart_df['price'].mean():,.0f}**
- Maximum Price : **${chart_df['price'].max():,.0f}**
- Minimum Price : **${chart_df['price'].min():,.0f}**
""")

    with right:

        st.success("""
### Model Used

✔ Random Forest Regressor

✔ Missing Value Handling

✔ One-Hot Encoding

✔ Feature Engineering

✔ Train-Test Split (80:20)

✔ Interactive Visualizations
""")
# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.title("🏠 Predict House Price")
    st.write("Enter the house details below to estimate its selling price.")

    prediction_df = df.copy()

    # Remove unused columns
    for col in ["date", "street"]:
        if col in prediction_df.columns:
            prediction_df.drop(columns=col, inplace=True)

    st.markdown("---")

    # --------------------------
    # Input Form
    # --------------------------

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=20,
                value=3
            )

            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5
            )

            sqft_living = st.number_input(
                "Living Area (sqft)",
                min_value=200,
                value=2000
            )

            sqft_lot = st.number_input(
                "Lot Area (sqft)",
                min_value=500,
                value=5000
            )

            floors = st.number_input(
                "Floors",
                min_value=1.0,
                max_value=5.0,
                value=1.0,
                step=0.5
            )

            waterfront = st.selectbox(
                "Waterfront",
                [0, 1]
            )

        with col2:

            view = st.slider(
                "View Rating",
                0,
                4,
                0
            )

            condition = st.slider(
                "Condition",
                1,
                5,
                3
            )

            sqft_above = st.number_input(
                "Above Ground Area",
                min_value=200,
                value=1500
            )

            sqft_basement = st.number_input(
                "Basement Area",
                min_value=0,
                value=500
            )

            yr_built = st.number_input(
                "Year Built",
                min_value=1900,
                max_value=2026,
                value=2000
            )

            yr_renovated = st.number_input(
                "Year Renovated (0 if never)",
                min_value=0,
                max_value=2026,
                value=0
            )

        # --------------------------
        # City Selection
        # --------------------------

        if "city" in prediction_df.columns:
            city = st.selectbox(
                "City",
                sorted(prediction_df["city"].unique())
            )
        else:
            city = ""

        if "statezip" in prediction_df.columns:
            statezip = st.selectbox(
                "State ZIP",
                sorted(prediction_df["statezip"].unique())
            )
        else:
            statezip = ""

        if "country" in prediction_df.columns:
            country = st.selectbox(
                "Country",
                sorted(prediction_df["country"].unique())
            )
        else:
            country = ""

        submitted = st.form_submit_button("🔮 Predict Price")

    # --------------------------
    # Prediction
    # --------------------------

    if submitted:

        input_data = pd.DataFrame({

            "bedrooms":[bedrooms],
            "bathrooms":[bathrooms],
            "sqft_living":[sqft_living],
            "sqft_lot":[sqft_lot],
            "floors":[floors],
            "waterfront":[waterfront],
            "view":[view],
            "condition":[condition],
            "sqft_above":[sqft_above],
            "sqft_basement":[sqft_basement],
            "yr_built":[yr_built],
            "yr_renovated":[yr_renovated],
            "city":[city],
            "statezip":[statezip],
            "country":[country]

        })

        prediction = model.predict(input_data)[0]

        st.markdown("---")

        st.success("Prediction Completed Successfully!")

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(90deg,#11998e,#38ef7d);
                padding:30px;
                border-radius:20px;
                text-align:center;
                color:white;
            ">

            <h2>🏠 Estimated House Price</h2>

            <h1>${prediction:,.2f}</h1>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()

        # --------------------------
        # House Summary
        # --------------------------

        st.markdown("## 📋 Property Summary")

        summary = pd.DataFrame({
            "Feature":[
                "Bedrooms",
                "Bathrooms",
                "Living Area",
                "Lot Area",
                "Floors",
                "Waterfront",
                "View",
                "Condition",
                "Above Ground",
                "Basement",
                "Year Built",
                "Renovated",
                "City"
            ],

            "Value":[
                bedrooms,
                bathrooms,
                sqft_living,
                sqft_lot,
                floors,
                waterfront,
                view,
                condition,
                sqft_above,
                sqft_basement,
                yr_built,
                yr_renovated,
                city
            ]
        })

        st.dataframe(summary, use_container_width=True)
# =====================================================
# BATCH PREDICTION
# =====================================================

elif page == "Batch Prediction":

    st.title("📂 Batch House Price Prediction")

    st.write("""
Upload a CSV file containing multiple house records.
The model will predict prices for all houses and allow you to download the results.
""")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(uploaded_file)

            st.success("✅ File Uploaded Successfully!")

            st.subheader("Preview of Uploaded Data")

            st.dataframe(batch_df.head(), use_container_width=True)

            st.markdown("---")

            # Remove unwanted columns if present
            for col in ["date", "street"]:
                if col in batch_df.columns:
                    batch_df.drop(columns=col, inplace=True)

            # Required columns
            required_columns = [
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "floors",
                "waterfront",
                "view",
                "condition",
                "sqft_above",
                "sqft_basement",
                "yr_built",
                "yr_renovated",
                "city",
                "statezip",
                "country"
            ]

            missing = [
                col for col in required_columns
                if col not in batch_df.columns
            ]

            if missing:

                st.error("❌ Missing Columns:")

                st.write(missing)

            else:

                if st.button("🚀 Predict All Houses"):

                    predictions = model.predict(batch_df)

                    result_df = batch_df.copy()

                    result_df["Predicted Price"] = predictions

                    st.success("✅ Prediction Completed!")

                    st.subheader("Prediction Results")

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    # -----------------------------
                    # Summary Metrics
                    # -----------------------------

                    st.markdown("---")

                    st.subheader("📊 Prediction Summary")

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Average Price",
                        f"${predictions.mean():,.0f}"
                    )

                    c2.metric(
                        "Highest Price",
                        f"${predictions.max():,.0f}"
                    )

                    c3.metric(
                        "Lowest Price",
                        f"${predictions.min():,.0f}"
                    )

                    # -----------------------------
                    # Price Distribution
                    # -----------------------------

                    st.markdown("---")

                    st.subheader("📈 Predicted Price Distribution")

                    fig = px.histogram(
                        result_df,
                        x="Predicted Price",
                        nbins=30,
                        color_discrete_sequence=["green"],
                        title="Predicted House Prices"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # -----------------------------
                    # Download Button
                    # -----------------------------

                    csv = result_df.to_csv(index=False)

                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="house_price_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(f"Error : {e}")

    else:

        st.info("Please upload a CSV file to begin.")
# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.title("📖 About This Project")

    st.markdown("""
# 🏠 House Price Prediction System

This web application predicts the selling price of a house using
Machine Learning and Streamlit.

The application was developed to help users estimate house prices
based on various property features such as bedrooms, bathrooms,
living area, lot size, location, and construction year.
""")

    st.markdown("---")

    # ------------------------------------------------
    # Project Features
    # ------------------------------------------------

    st.subheader("✨ Features")

    st.markdown("""
- 📊 Interactive Dashboard
- 📈 Data Visualizations
- 🤖 Machine Learning Prediction
- 📂 Batch CSV Prediction
- 📥 Download Prediction Results
- 📋 Dataset Statistics
- 📉 Correlation Heatmap
- 📊 Model Performance Metrics
- 💻 User-Friendly Interface
""")

    st.markdown("---")

    # ------------------------------------------------
    # Machine Learning
    # ------------------------------------------------

    st.subheader("🤖 Machine Learning Model")

    st.info("""
### Algorithm Used

**Random Forest Regressor**

Random Forest is an ensemble learning algorithm that combines
multiple Decision Trees to improve prediction accuracy.

Advantages:

- High Accuracy
- Handles Large Datasets
- Reduces Overfitting
- Works Well with Numerical and Categorical Data
- Robust and Reliable
""")

    st.markdown("---")

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    st.subheader("📂 Dataset Information")

    rows, cols = df.shape

    st.write(f"**Total Rows :** {rows:,}")
    st.write(f"**Total Columns :** {cols}")
    st.write(f"**Target Variable :** Price")

    st.markdown("---")

    # ------------------------------------------------
    # Technologies
    # ------------------------------------------------

    st.subheader("🛠 Technologies Used")

    tech = pd.DataFrame({

        "Technology":[
            "Python",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Scikit-Learn",
            "Plotly",
            "Pickle"
        ],

        "Purpose":[
            "Programming Language",
            "Web Application",
            "Data Processing",
            "Numerical Computation",
            "Machine Learning",
            "Interactive Charts",
            "Model Saving"
        ]

    })

    st.table(tech)

    st.markdown("---")

    # ------------------------------------------------
    # Performance
    # ------------------------------------------------

    st.subheader("📊 Model Performance")

    c1, c2 = st.columns(2)

    with c1:

        st.metric("R² Score", f"{metrics['R2']:.3f}")

        st.metric("MAE", f"{metrics['MAE']:,.2f}")

    with c2:

        st.metric("RMSE", f"{metrics['RMSE']:,.2f}")

        st.metric("MSE", f"{metrics['MSE']:,.2f}")

    st.markdown("---")

    # ------------------------------------------------
    # Workflow
    # ------------------------------------------------

    st.subheader("⚙️ Project Workflow")

    st.markdown("""
1️⃣ Load Dataset

⬇️

2️⃣ Data Cleaning

⬇️

3️⃣ Feature Engineering

⬇️

4️⃣ Train-Test Split

⬇️

5️⃣ Train Random Forest Model

⬇️

6️⃣ Evaluate Model

⬇️

7️⃣ Save Model

⬇️

8️⃣ Streamlit Deployment

⬇️

9️⃣ User Prediction
""")

    st.markdown("---")

    # ------------------------------------------------
    # Footer
    # ------------------------------------------------

    st.success("""
🎉 Thank you for using the House Price Prediction System!

Developed using ❤️ Python, Streamlit, Plotly, and Scikit-Learn.
""")