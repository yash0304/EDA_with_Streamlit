📊 Streamlit Exploratory Data Analysis (EDA) App

This project is an interactive EDA and ML app built with Streamlit.

It allows you to:

✅ Upload your own CSV file

✅ Preview and clean data (numeric-only filtering, column removal)

✅ Generate profiling reports using ydata-profiling

✅ Select a target column and train a Random Forest Classifier

✅ View model performance (accuracy & classification report)

✅ Create quick visualizations (scatter, line, bar plots)



**🚀 Features**

Data Upload: Upload any CSV file

Data Cleaning: Remove unnecessary columns, filter numeric-only data.

Profiling Report: Full profiling using ydata-profiling.

ML Model: Train a Random Forest Classifier on your selected target column.

Visualizations: Generate quick exploratory plots.

**📦 Installation**

Clone the repository:

git clone https://github.com/yash0304/EDA_with_Streamlit.git

cd eda-app

Create and activate a virtual environment (recommended):

python -m venv venv

source venv/bin/activate   # for Linux/Mac

venv\Scripts\activate      # for Windows

**Install dependencies:**

pip install -r requirements.txt


**▶️ Running the App**

Start the Streamlit server:

streamlit run streamlitEDA.py

Then open the app in your browser at:

http://localhost:8501

**📂 Project Structure**

eda-app/

  │── streamlitEDA.py      # Main Streamlit app

  │── requirements.txt     # Dependencies  

  │── README.md            # Documentation


**📋 Example Workflow**

Upload your dataset (CSV).

Preview first rows.

(Optional) Keep only numeric columns.

Generate profiling report.

Remove unnecessary columns.

Select a target column for ML.

Train Random Forest Classifier → view accuracy & classification report.

Generate custom plots.


**⚡ Requirements**

Main dependencies:

streamlit

pandas

numpy

matplotlib

scikit-learn

ydata-profiling

streamlit-pandas-profiling

(See requirements.txt for details)
