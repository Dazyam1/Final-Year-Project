import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MedPredict AI",
    page_icon="🏥",
    layout="wide"
)

# Simplified CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin: 0;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .positive-prediction {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    
    .negative-prediction {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    
    .prediction-box h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .disease-header {
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        position: relative;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    .disease-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 15px;
    }
    
    .disease-header h2 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
    }
    
    .hepatitis-bg {
        background-image: url('https://media.sciencephoto.com/image/f0121049/800wm/F0121049-Human_liver_with_hepatitis_viruses.jpg');
    }
    
    .tb-bg {
        background-image: url('https://c7.alamy.com/comp/2CTWJPT/patient-friendly-scheme-of-tb-damages-in-human-lung-anatomical-diagram-of-tuberculosis-respiratory-system-diseases-2CTWJPT.jpg');
    }
    
    .hiv-bg {
        background-image: url('https://www.std-gov.org/images/std-gov-2775963-hiv-virus-l-640x600.jpg');
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_disease' not in st.session_state:
    st.session_state.selected_disease = "Hepatitis"

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 MedPredict AI</h1>
    <p>Multi-Disease Prediction Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Disease Selection")
    diseases = ["Hepatitis", "Tuberculosis (TB)", "HIV"]
    for disease in diseases:
        if st.button(disease, key=f"nav_{disease}", use_container_width=True):
            st.session_state.selected_disease = disease
    
    st.markdown("---")
    st.warning("**DISCLAIMER:** For educational purposes only. Always consult healthcare professionals.")

def load_model_safe(filename, model_type="joblib"):
    """Safely load models with error handling"""
    try:
        if model_type == "joblib":
            return joblib.load(filename)
        else:
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file '{filename}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

def hepatitis_prediction():
    st.markdown("""
    <div class="disease-header hepatitis-bg">
        <h2>🫁 Hepatitis Prognosis Prediction</h2>
    </div>
    """, unsafe_allow_html=True)
    
    hepatitis_model = load_model_safe('hepatitis_model.pkl', 'pickle')
    if hepatitis_model is None:
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Demographics
        st.subheader("Patient Information")
        age = st.slider("Age", 7, 78, 40)
        sex = st.selectbox("Sex", ['male', 'female'])
        
        # Treatment
        st.subheader("Treatment History")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            steroid = st.selectbox("Steroid Treatment", ['False', 'True', 'Unknown'])
        with col_t2:
            antivirals = st.selectbox("Antiviral Treatment", ['False', 'True'])
        
        # Symptoms
        st.subheader("Symptoms")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fatigue = st.selectbox("Fatigue", ['False', 'True', 'Unknown'])
            malaise = st.selectbox("Malaise", ['False', 'True', 'Unknown'])
            anorexia = st.selectbox("Anorexia", ['False', 'True', 'Unknown'])
            liver_big = st.selectbox("Enlarged Liver", ['False', 'True', 'Unknown'])
            liver_firm = st.selectbox("Firm Liver", ['False', 'True', 'Unknown'])
        with col_s2:
            spleen_palpable = st.selectbox("Palpable Spleen", ['False', 'True', 'Unknown'])
            spiders = st.selectbox("Spider Angiomas", ['False', 'True', 'Unknown'])
            ascites = st.selectbox("Ascites", ['False', 'True', 'Unknown'])
            varices = st.selectbox("Varices", ['False', 'True', 'Unknown'])
            histology = st.selectbox("Histology", ['False', 'True'])
        
        # Lab Results
        st.subheader("Lab Results")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            bilirubin = st.number_input("Bilirubin (mg/dL)", 0.3, 8.0, 1.0, 0.1)
            alk_phosphate = st.number_input("Alkaline Phosphatase", 26, 295, 85)
            sgot = st.number_input("SGOT/AST", 14, 648, 25)
        with col_l2:
            albumin = st.number_input("Albumin (g/dL)", 2.1, 6.4, 4.0, 0.1)
            protime = st.number_input("Prothrombin Time (%)", 0, 100, 85)
    
    with col2:
        st.markdown("### Prediction")
        
        if st.button("🔍 Predict", type="primary"):
            # Helper functions
            def map_bool_str_to_int(val):
                if val == 'True': return 1
                elif val == 'False': return 0
                else: return -1
            
            def map_sex_to_int(val):
                return 0 if val == 'male' else 1
            
            # Prepare input data
            input_data = [
                int(age), map_sex_to_int(sex),
                map_bool_str_to_int(steroid), map_bool_str_to_int(antivirals),
                map_bool_str_to_int(fatigue), map_bool_str_to_int(malaise),
                map_bool_str_to_int(anorexia), map_bool_str_to_int(liver_big),
                map_bool_str_to_int(liver_firm), map_bool_str_to_int(spleen_palpable),
                map_bool_str_to_int(spiders), map_bool_str_to_int(ascites),
                map_bool_str_to_int(varices), map_bool_str_to_int(histology),
                float(bilirubin), float(alk_phosphate), float(sgot),
                float(albumin), float(protime)
            ]
            
            try:
                input_array = np.array(input_data).reshape(1, -1)
                prediction = hepatitis_model.predict(input_array)[0]
                proba = hepatitis_model.predict_proba(input_array)[0]
                
                live_prob = proba[1] if 1 in hepatitis_model.classes_ else proba[0]
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box positive-prediction">
                        <h2>✅ POSITIVE PROGNOSIS</h2>
                        <h3>Patient Likely to Survive</h3>
                        <p>Confidence: {live_prob*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box negative-prediction">
                        <h2>⚠️ CONCERNING PROGNOSIS</h2>
                        <h3>Requires Attention</h3>
                        <p>Survival Confidence: {live_prob*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def tb_prediction():
    st.markdown("""
    <div class="disease-header tb-bg">
        <h2>🦠 Tuberculosis Risk Assessment</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tb_model = load_model_safe('tb_predictor_model.pkl')
    if tb_model is None:
        return
    
    feature_names = [
        "fever for two weeks", "coughing blood", "sputum mixed with blood",
        "night sweats", "chest pain", "back pain in certain parts",
        "shortness of breath", "weight loss", "body feels tired",
        "lumps that appear around the armpits and neck",
        "cough and phlegm continuously for two weeks to four weeks",
        "swollen lymph nodes", "loss of appetite"
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("TB Symptom Assessment")
        user_input = []
        
        for i, symptom in enumerate(feature_names):
            val = st.selectbox(f"{symptom.title()}?", ["No", "Yes"], key=f"tb_{i}")
            user_input.append(1 if val == "Yes" else 0)
    
    with col2:
        st.markdown("### Risk Assessment")
        
        if st.button("🔍 Assess TB Risk", type="primary"):
            try:
                prediction = tb_model.predict([user_input])[0]
                proba = tb_model.predict_proba([user_input])[0]
                confidence = proba[prediction]
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box negative-prediction">
                        <h2>⚠️ HIGH TB RISK</h2>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Consult a healthcare provider immediately for TB testing.</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box positive-prediction">
                        <h2>✅ LOW TB RISK</h2>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def hiv_prediction():
    st.markdown("""
    <div class="disease-header hiv-bg">
        <h2>🩸 HIV Risk Assessment</h2>
    </div>
    """, unsafe_allow_html=True)
    
    hiv_model = load_model_safe('hiv_model.pkl')
    vectorizer = load_model_safe('vectorizer.pkl')
    
    if hiv_model is None or vectorizer is None:
        return
    
    symptoms_list = [
        "Fever", "Night sweats", "Fatigue", "Weight loss", "Persistent diarrhea",
        "Swollen lymph nodes", "Skin rashes", "Oral thrush", "Memory loss",
        "Neurological disorders", "Opportunistic infections", "CD4 count low",
        "Viral load high", "Frequent infections"
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("HIV Symptom Checklist")
        selected_symptoms = []
        
        for symptom in symptoms_list:
            if st.checkbox(symptom, key=f"hiv_{symptom}"):
                selected_symptoms.append(symptom)
    
    with col2:
        st.markdown("### Risk Assessment")
        
        if st.button("🔍 Assess HIV Risk", type="primary"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
                return
            
            try:
                symptoms_text = ", ".join(selected_symptoms)
                text_vec = vectorizer.transform([symptoms_text])
                prediction = hiv_model.predict(text_vec)[0]
                probability = hiv_model.predict_proba(text_vec)[0]
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box negative-prediction">
                        <h2>⚠️ HIV-RELATED SYMPTOMS</h2>
                        <p>HIV Probability: {probability[1]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Consult a healthcare provider for HIV testing.</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box positive-prediction">
                        <h2>✅ LOW HIV RISK</h2>
                        <p>HIV Probability: {probability[1]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write(f"**Selected symptoms:** {', '.join(selected_symptoms)}")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# Main application
def main():
    # Disease selection
    st.markdown("### 🎯 Select Disease for Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🫁 Hepatitis", use_container_width=True):
            st.session_state.selected_disease = "Hepatitis"
    
    with col2:
        if st.button("🦠 Tuberculosis", use_container_width=True):
            st.session_state.selected_disease = "Tuberculosis (TB)"
    
    with col3:
        if st.button("🩸 HIV", use_container_width=True):
            st.session_state.selected_disease = "HIV"
    
    st.markdown("---")
    
    # Display selected disease prediction
    if st.session_state.selected_disease == "Hepatitis":
        hepatitis_prediction()
    elif st.session_state.selected_disease == "Tuberculosis (TB)":
        tb_prediction()
    elif st.session_state.selected_disease == "HIV":
        hiv_prediction()

if __name__ == "__main__":
    main()