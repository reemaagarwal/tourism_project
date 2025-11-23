import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# CONFIGURATION
# -----------------------------

HF_MODEL_REPO = "Reemaagarwal/visit-with-us-tourism-model"
MODEL_FILENAME = "best_model.joblib"

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    """
    Download the trained model from the Hugging Face Model Hub
    and load it into memory.
    """
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILENAME
    )
    model = joblib.load(model_path)
    return model

model = load_model()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Wellness Tourism Package Purchase Predictor")
st.write(
    """
    This app predicts whether a customer is likely to purchase the 
    **Wellness Tourism Package** based on their profile and interaction details.
    """
)

st.markdown("### 🧍 Customer Information")

# Numeric and categorical inputs
age = st.number_input("Age", min_value=18, max_value=80, value=30)
city_tier = st.selectbox("City Tier", options=[1, 2, 3], index=0)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=300, value=30)
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
preferred_star = st.selectbox("Preferred Property Star", options=[1,2,3,4,5], index=2)
num_trips = st.number_input("Number of Trips per year", min_value=0, max_value=30, value=2)
passport = st.selectbox("Passport (0 = No, 1 = Yes)", options=[0,1], index=1)
pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=4)
own_car = st.selectbox("Own Car (0 = No, 1 = Yes)", options=[0,1], index=0)
num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=50000)

st.markdown("### 🧾 Categorical Details")

type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Freelancer", "Others"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox(
    "Designation", 
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

# -----------------------------
# BUILD INPUT DATAFRAME
# -----------------------------
input_data = {
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "PreferredPropertyStar": preferred_star,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "TypeofContact": type_of_contact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status,
    "Designation": designation
}

input_df = pd.DataFrame([input_data])

st.markdown("### 📋 Input Summary")
st.dataframe(input_df)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Purchase Probability"):
    # Model pipeline handles preprocessing internally
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # probability of class "1" (purchase)

    st.markdown("### 🔮 Prediction Result")
    if pred == 1:
        st.success(f"The model predicts that the customer is **likely to purchase** the package.")
    else:
        st.info(f"The model predicts that the customer is **unlikely to purchase** the package.")

    st.write(f"Estimated probability of purchase: **{proba:.2f}**")
