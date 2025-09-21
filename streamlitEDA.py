import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Exploratory Data Analysis (EDA) App")

uploaded_fi = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_fi is not None:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_fi)
    df = st.session_state.df
    st.subheader("Data Preview")
    st.dataframe(df.head())

    #keeping only numeric columns
    if st.checkbox("If you want only numeric columns"):
        st.session_state.df = df.select_dtypes(include=[np.number])
        df = st.session_state.df
        st.dataframe(df.head())

    #Profiling report
    with st.expander("Show profiling repport"):
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)

    #removing unnecessary columns
    
    if st.checkbox("Remove Unnecessary Columns"):
            all_columns = df.columns.tolist()
            remove_columns = st.multiselect("Select columns to remove", all_columns)

            if remove_columns:
                 st.session_state.df = df.drop(columns=remove_columns)
                 df = st.session_state.df
                 st.write("You selected to remove: ", remove_columns)
                 st.dataframe(df.head())
                 st.write(f"Columns removed successfully: {remove_columns}")

    #target selection
    if st.checkbox("Select Target Column"):
            target = st.selectbox("select any target column",df.columns)       
            st.write(f"You selected {target} as target column")
        

        #model training
            if st.button("train model"):
                st.session_state.df = df.dropna()
                df = st.session_state.df
                 # Encode categorical variables if any
                if df[target].dtype == 'object':
                    df[target] = df[target].astype('category').cat.codes
                    
                X = df.drop(columns=[target])
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test,y_pred)
                st.write(f"Model Accuracy: {acc}")
                st.text("Classification Report")
                st.text(classification_report(y_test,y_pred))
        
    #visualizations
    if st.checkbox("Show Plots"):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col1 = st.selectbox("Select X-axis", num_cols)
        col2 = st.selectbox("Select Y-axis", num_cols)
        plot_type = st.selectbox("Select Plot Type", ["scatter","line","bar"])
        if st.button("Generate Plot"):
            if plot_type == "line":
                plt.plot(df[col1],df[col2])
                st.pyplot(plt)
            elif plot_type == "scatter":
                plt.scatter(df[col1],df[col2])
                st.pyplot(plt)
            elif plot_type == "bar":
                plt.bar(df[col1],df[col2])
                st.pyplot(plt)