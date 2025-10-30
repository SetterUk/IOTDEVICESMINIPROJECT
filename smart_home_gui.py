#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="Smart Home Efficiency Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_preprocessor():
    """Load trained models and preprocessor"""
    try:
        models = joblib.load('trained_models.joblib')
        preprocessor = joblib.load('smart_home_preprocessor.joblib')
        model_comparison = pd.read_csv('model_comparison.csv', index_col=0)
        return models, preprocessor, model_comparison
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def get_feature_names():
    """Get feature names from the original data structure"""
    # Based on the preprocessing script, these are the expected features
    return [
        'UserID', 'DeviceType', 'UsageHoursPerDay', 'EnergyConsumption',
        'MalfunctionIncidents', 'UserPreferences'
    ]

def create_input_form():
    """Create input form for device parameters"""
    st.sidebar.markdown('<div class="sidebar-header">📝 Device Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns(2)

    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=1000, value=1, step=1)
        device_type = st.selectbox("Device Type",
                                   ["Refrigerator", "Washing Machine", "Air Conditioner",
                                    "Television", "Microwave", "Dishwasher", "Laptop",
                                    "Smartphone", "Router", "Security Camera"])
        usage_hours = st.slider("Usage Hours Per Day", 0.0, 24.0, 8.0, 0.5)

    with col2:
        energy_consumption = st.slider("Energy Consumption (kWh)", 0.0, 50.0, 5.0, 0.1)
        malfunction_incidents = st.slider("Malfunction Incidents", 0, 10, 0, 1)
        user_preferences = st.selectbox("User Preferences",
                                       ["Eco-friendly", "Performance", "Cost-saving", "Convenience"])

    return {
        'UserID': user_id,
        'DeviceType': device_type,
        'UsageHoursPerDay': usage_hours,
        'EnergyConsumption': energy_consumption,
        'MalfunctionIncidents': malfunction_incidents,
        'UserPreferences': user_preferences
    }

def make_prediction(input_data, model, preprocessor):
    """Make prediction using the model"""
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input
    try:
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0]

        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def display_prediction(prediction, prediction_proba, confidence_threshold=0.8):
    """Display prediction results"""
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("### 🔮 Prediction Results")

    if prediction == 1:
        st.success("✅ **Efficient Device**")
        efficiency_status = "Efficient"
    else:
        st.error("❌ **Inefficient Device**")
        efficiency_status = "Inefficient"

    confidence = max(prediction_proba) * 100
    st.metric("Confidence Score", f"{confidence:.1f}%")

    # Confidence gauge
    if confidence >= confidence_threshold * 100:
        st.success(f"High confidence prediction (>{confidence_threshold*100:.0f}%)")
    else:
        st.warning(f"Low confidence prediction (<={confidence_threshold*100:.0f}%)")

    st.markdown('</div>', unsafe_allow_html=True)

    return efficiency_status, confidence

def create_model_dashboard(model_comparison):
    """Create model performance dashboard"""
    st.markdown("### 📊 Model Performance Dashboard")

    # Best model highlight
    best_model = model_comparison['f1'].idxmax()
    best_f1 = model_comparison['f1'].max()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", best_model)
    with col2:
        st.metric("Best F1-Score", f"{best_f1:.4f}")
    with col3:
        st.metric("Total Models", len(model_comparison))
    with col4:
        st.metric("Accuracy Range", f"{model_comparison['accuracy'].min():.3f} - {model_comparison['accuracy'].max():.3f}")

    # Performance comparison chart
    fig = px.bar(model_comparison.reset_index(),
                 x='index', y=['accuracy', 'precision', 'recall', 'f1'],
                 title="Model Performance Comparison",
                 barmode='group',
                 labels={'index': 'Model', 'value': 'Score'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ROC AUC comparison
    fig2 = px.bar(model_comparison.reset_index(),
                  x='index', y='roc_auc',
                  title="ROC AUC Scores",
                  color='roc_auc',
                  color_continuous_scale='viridis')
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

def create_feature_importance_plot(models):
    """Create feature importance visualization for Decision Tree"""
    st.markdown("### 🎯 Feature Importance Analysis")

    dt_model = models.get('Decision Tree')
    if dt_model and hasattr(dt_model, 'feature_importances_'):
        # Get feature names from preprocessor
        try:
            feature_names = get_feature_names()
            importances = dt_model.feature_importances_

            # Create DataFrame for plotting
            feat_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)

            fig = px.bar(feat_imp_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Decision Tree Feature Importance",
                        color='importance',
                        color_continuous_scale='blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating feature importance plot: {e}")
    else:
        st.warning("Feature importance not available for Decision Tree model")

def batch_prediction_interface(models, preprocessor):
    """Batch prediction interface"""
    st.markdown("### 📤 Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file with device data", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())

            if st.button("Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    # Make predictions
                    dt_model = models['Decision Tree']
                    batch_processed = preprocessor.transform(batch_data)
                    predictions = dt_model.predict(batch_processed)
                    probabilities = dt_model.predict_proba(batch_processed)

                    # Add results to dataframe
                    batch_data['Predicted_Efficiency'] = predictions
                    batch_data['Confidence'] = np.max(probabilities, axis=1)
                    batch_data['Efficiency_Status'] = batch_data['Predicted_Efficiency'].map({1: 'Efficient', 0: 'Inefficient'})

                    st.success("Batch prediction completed!")
                    st.dataframe(batch_data)

                    # Download results
                    csv = batch_data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">Download Predictions CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing batch file: {e}")

def export_single_prediction(input_data, prediction, confidence, efficiency_status):
    """Export single prediction results"""
    st.markdown("### 💾 Export Results")

    result_df = pd.DataFrame([{
        **input_data,
        'Predicted_Efficiency': prediction,
        'Efficiency_Status': efficiency_status,
        'Confidence_Score': confidence
    }])

    st.dataframe(result_df)

    csv = result_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="single_prediction.csv">Download Prediction CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Smart Home Device Efficiency Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict whether a smart home device is efficient or inefficient using machine learning models.")

    # Load models
    models, preprocessor, model_comparison = load_models_and_preprocessor()

    if models is None or preprocessor is None:
        st.error("Failed to load models and preprocessor. Please ensure the required files are present.")
        return

    # Navigation
    page = st.sidebar.selectbox("Navigation",
                               ["Single Prediction", "Model Dashboard", "Feature Importance", "Batch Prediction"])

    if page == "Single Prediction":
        # Input form
        input_data = create_input_form()

        # Prediction button
        if st.sidebar.button("🔮 Make Prediction", type="primary"):
            with st.spinner("Analyzing device efficiency..."):
                dt_model = models['Decision Tree']
                prediction, prediction_proba = make_prediction(input_data, dt_model, preprocessor)

                if prediction is not None:
                    efficiency_status, confidence = display_prediction(prediction, prediction_proba)

                    # Export option
                    export_single_prediction(input_data, prediction, confidence, efficiency_status)

    elif page == "Model Dashboard":
        create_model_dashboard(model_comparison)

    elif page == "Feature Importance":
        create_feature_importance_plot(models)

    elif page == "Batch Prediction":
        batch_prediction_interface(models, preprocessor)

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit | Using Decision Tree Model for Predictions")

if __name__ == "__main__":
    main()