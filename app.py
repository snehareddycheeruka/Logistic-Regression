import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="AI Student Marks Predictor",
    page_icon="📚",
    layout="wide"
)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<h1 style='text-align:center; color:#4B8BBE;'>
📚 AI Student Marks Predictor
</h1>

<p style='text-align:center; font-size:18px;'>
Predict student final marks using Machine Learning
</p>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    data = pd.read_csv("student_data_set.csv")
    return data

df = load_data()

# ==========================================================
# CLEAN DATA
# ==========================================================

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop("Final_Marks", axis=1)
y = df["Final_Marks"]

# ==========================================================
# SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=10
)

# ==========================================================
# SCALING
# ==========================================================

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# MODEL TRAINING
# ==========================================================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================================
# MODEL SCORE
# ==========================================================

predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Prediction",
        "Dataset",
        "Charts",
        "Model Info"
    ]
)

# ==========================================================
# PREDICTION PAGE
# ==========================================================

if page == "Prediction":

    st.subheader("🧠 Enter Student Details")

    c1, c2 = st.columns(2)

    with c1:

        hours = st.number_input(
            "Hours Studied",
            1,
            24,
            5
        )

        attendance = st.slider(
            "Attendance %",
            0,
            100,
            80
        )

        assignments = st.slider(
            "Assignment Score",
            0,
            100,
            70
        )

        sleep = st.slider(
            "Sleep Hours",
            1,
            12,
            7
        )

    with c2:

        previous = st.slider(
            "Previous Marks",
            0,
            100,
            60
        )

        internet = st.slider(
            "Internet Usage",
            0,
            24,
            4
        )

        activities = st.slider(
            "Activities",
            0,
            10,
            3
        )

        mock = st.slider(
            "Mock Test Score",
            0,
            10,
            5
        )

    efficiency = hours / (internet + 1)

    if st.button("🎯 Predict Marks"):

        sample = pd.DataFrame({
            "Hours_Studied": [hours],
            "Attendance": [attendance],
            "Assignments": [assignments],
            "Sleep_Hours": [sleep],
            "Previous_Marks": [previous],
            "Internet_Usage_Hours": [internet],
            "Extracurricular_Activities": [activities],
            "Mock_Test_Score": [mock]
            
        })

        sample_scaled = scaler.transform(sample)

        result = model.predict(sample_scaled)[0]

        st.success(
            f"Predicted Final Marks: {result:.2f}"
        )

        progress = int(result)

        st.progress(
            min(progress, 100)
        )

        # Grade Section
        if result >= 90:
            st.info("Grade: A+ 🌟")

        elif result >= 75:
            st.info("Grade: A 👍")

        elif result >= 60:
            st.info("Grade: B 📘")

        elif result >= 40:
            st.warning("Grade: C ⚠️")

        else:
            st.error("Grade: Fail ❌")

# ==========================================================
# DATASET PAGE
# ==========================================================

elif page == "Dataset":

    st.subheader("📂 Student Dataset")

    st.dataframe(df)

    st.write("Shape of Dataset:", df.shape)

# ==========================================================
# CHARTS PAGE
# ==========================================================

elif page == "Charts":

    st.subheader("📊 Data Visualizations")

    option = st.selectbox(
        "Choose Chart",
        [
            "Final Marks Distribution",
            "Study Hours vs Marks",
            "Attendance vs Marks"
        ]
    )

    # Histogram
    if option == "Final Marks Distribution":

        fig, ax = plt.subplots()

        ax.hist(df["Final_Marks"], bins=12)

        ax.set_title("Final Marks Distribution")

        st.pyplot(fig)

    # Scatter Plot
    elif option == "Study Hours vs Marks":

        fig, ax = plt.subplots()

        ax.scatter(
            df["Hours_Studied"],
            df["Final_Marks"]
        )

        ax.set_xlabel("Hours Studied")
        ax.set_ylabel("Final Marks")

        st.pyplot(fig)

    # Attendance Chart
    else:

        fig, ax = plt.subplots()

        ax.scatter(
            df["Attendance"],
            df["Final_Marks"]
        )

        ax.set_xlabel("Attendance")
        ax.set_ylabel("Final Marks")

        st.pyplot(fig)

# ==========================================================
# MODEL INFO PAGE
# ==========================================================

else:

    st.subheader("🤖 Model Details")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Algorithm",
            "Linear Regression"
        )

    with m2:
        st.metric(
            "Accuracy Score",
            f"{score:.2f}"
        )

    with m3:
        st.metric(
            "Rows",
            df.shape[0]
        )

    st.markdown("""
    ### 📖 About Model

    This project uses Linear Regression
    to predict student marks based on:

    - Study Hours
    - Attendance
    - Previous Marks
    - Sleep Hours
    - Mock Test Scores
    - Internet Usage

    The application is built using:
    - Streamlit
    - Pandas
    - Scikit-learn
    """)