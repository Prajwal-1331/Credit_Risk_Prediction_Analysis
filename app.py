import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Risk Analysis", layout="wide")

st.title("💳 Credit Risk Analysis Dashboard")

# Upload files
application_file = st.file_uploader(
    "Upload application_record.csv",
    type=["csv"]
)

credit_file = st.file_uploader(
    "Upload credit_record.csv",
    type=["csv"]
)

if application_file and credit_file:

    de = pd.read_csv(application_file)
    dt = pd.read_csv(credit_file)

    # Merge
    df = pd.merge(de, dt, on='ID')

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Rename columns
    df.rename(columns={
        'AMT_INCOME_TOTAL': 'Annual_Income',
        'CNT_FAM_MEMBERS': 'Family_Member_Count',
        'DAYS_BIRTH': 'Age_Days',
        'DAYS_EMPLOYED': 'Employment_Days',
        'NAME_INCOME_TYPE': 'Income_Type',
        'OCCUPATION_TYPE': 'Occupation_Type'
    }, inplace=True)

    # Fill missing values
    df['Occupation_Type'] = df['Occupation_Type'].fillna('Laborers')

    # Create Credit Risk
    df['Credit_Risk'] = df['STATUS']

    df['Credit_Risk'] = df['Credit_Risk'].replace(
        ['2', '3', '4', '5'], 1
    )

    df['Credit_Risk'] = df['Credit_Risk'].replace(
        ['1', '0', 'C', 'X'], 0
    )

    df['Credit_Risk'] = df['Credit_Risk'].astype(int)

    # Convert days to years
    df['Age_Days'] = abs(df['Age_Days']) // 365
    df['Employment_Days'] = abs(df['Employment_Days']) // 365

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # ======================
    # EDA Section
    # ======================

    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sb.countplot(x=df['Credit_Risk'], ax=ax)
        ax.set_title("Credit Risk Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sb.countplot(x=df['STATUS'], ax=ax)
        ax.set_title("STATUS Distribution")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        sb.histplot(df['Annual_Income'], ax=ax)
        ax.set_title("Annual Income Distribution")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        sb.histplot(df['Age_Days'], ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    st.subheader("Employment Length Distribution")

    fig, ax = plt.subplots()
    sb.histplot(df['Employment_Days'], bins=15, ax=ax)
    st.pyplot(fig)

    st.subheader("Family Member Count")

    fig, ax = plt.subplots()
    sb.countplot(x=df['Family_Member_Count'], ax=ax)
    st.pyplot(fig)

    st.subheader("Income Type vs Credit Risk")

    fig, ax = plt.subplots(figsize=(10, 5))
    sb.barplot(
        x=df['Income_Type'],
        y=df['Credit_Risk'],
        ax=ax
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Occupation Type vs Credit Risk")

    fig, ax = plt.subplots(figsize=(15, 5))
    sb.barplot(
        x=df['Occupation_Type'],
        y=df['Credit_Risk'],
        ax=ax
    )
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # ======================
    # Feature Engineering
    # ======================

    model_df = df.copy()

    model_df = model_df.drop([
        'ID',
        'STATUS',
        'CODE_GENDER',
        'CNT_CHILDREN',
        'Age_Days',
        'FLAG_MOBIL',
        'FLAG_WORK_PHONE',
        'FLAG_PHONE',
        'FLAG_EMAIL'
    ], axis=1)

    label = LabelEncoder()

    categorical_cols = [
        'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY',
        'Income_Type',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'Occupation_Type'
    ]

    for col in categorical_cols:
        model_df[col] = label.fit_transform(
            model_df[col]
        )

    X = model_df.drop(
        'Credit_Risk',
        axis=1
    )

    y = model_df['Credit_Risk']

    xtrain, xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ======================
    # Model Training
    # ======================

    st.header("🤖 Machine Learning Models")

    if st.button("Train Models"):

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(xtrain, ytrain)

        pred_lr = lr.predict(xtest)

        acc_lr = accuracy_score(
            ytest,
            pred_lr
        )

        # Random Forest
        rf = RandomForestClassifier(
            random_state=42
        )

        rf.fit(xtrain, ytrain)

        pred_rf = rf.predict(xtest)

        acc_rf = accuracy_score(
            ytest,
            pred_rf
        )

        # XGBoost
        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss'
        )

        xgb.fit(xtrain, ytrain)

        pred_xgb = xgb.predict(xtest)

        acc_xgb = accuracy_score(
            ytest,
            pred_xgb
        )

        results = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost"
            ],
            "Accuracy (%)": [
                round(acc_lr * 100, 2),
                round(acc_rf * 100, 2),
                round(acc_xgb * 100, 2)
            ]
        })

        st.subheader("Model Performance")
        st.dataframe(results)

        best_model = results.loc[
            results["Accuracy (%)"].idxmax()
        ]

        st.success(
            f"Best Model: {best_model['Model']} "
            f"({best_model['Accuracy (%)']}%)"
        )

        fig, ax = plt.subplots()

        sb.barplot(
            x="Model",
            y="Accuracy (%)",
            data=results,
            ax=ax
        )

        plt.xticks(rotation=20)
        st.pyplot(fig)
