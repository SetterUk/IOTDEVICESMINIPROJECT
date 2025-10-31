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
        model_comparison = pd.read_csv('model_comparison.csv', index_col=0)
        return models, model_comparison
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def get_feature_names():
    """Get actual feature names for visualization"""
    return [
        'UserID', 'UsageHoursPerDay', 'EnergyConsumption', 'UserPreferences', 
        'MalfunctionIncidents', 'DeviceAgeMonths', 'DeviceType_Camera', 
        'DeviceType_Lights', 'DeviceType_Security System', 'DeviceType_Smart Speaker', 
        'DeviceType_Thermostat'
    ]

def create_input_form():
    """Create input form for device parameters with real feature names"""
    st.sidebar.markdown('<div class="sidebar-header">📱 Smart Home Device Configuration</div>', unsafe_allow_html=True)
    
    # Device Type Selection
    st.sidebar.markdown("**🏠 Device Type**")
    device_type = st.sidebar.selectbox(
        "Select Device Type",
        options=['Smart Speaker', 'Camera', 'Lights', 'Security System', 'Thermostat'],
        help="Choose the type of smart home device"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Device Parameters**")
    
    col1, col2 = st.sidebar.columns(2)

    with col1:
        user_id = st.slider("👤 User ID", 1, 100, 5, 1, help="User identifier (1-100)")
        usage_hours = st.slider("⏰ Usage Hours/Day", 0.0, 24.0, 12.0, 0.5, help="Hours device is used per day (0-24)")
        energy_consumption = st.slider("⚡ Energy Consumption", 0.0, 10.0, 5.0, 0.1, help="Energy consumption level (0-10)")
        user_prefs = st.selectbox("✨ User Preferences", [0, 1], format_func=lambda x: "Low" if x == 0 else "High", help="User preference setting")

    with col2:
        malfunction_incidents = st.slider("🔧 Malfunction Incidents", 0, 10, 2, 1, help="Number of malfunctions (0-10)")
        device_age = st.slider("📅 Device Age (Months)", 0, 60, 24, 1, help="Device age in months (0-60)")
        
        st.markdown("*Normalized Feature Values:*")
        st.info(f"📌 Device: **{device_type}**", icon="ℹ️")

    # Convert to normalized features (0-10)
    # Note: These are approximate normalized values. The preprocessor normalizes these.
    # We're creating a simple representation for the model.
    
    # Create one-hot encoding for device type
    device_type_mapping = {
        'Camera': [1, 0, 0, 0, 0],
        'Lights': [0, 1, 0, 0, 0],
        'Security System': [0, 0, 1, 0, 0],
        'Smart Speaker': [0, 0, 0, 1, 0],
        'Thermostat': [0, 0, 0, 0, 1]
    }
    
    device_encoding = device_type_mapping[device_type]
    
    return {
        '0': (user_id - 50.5) / 29.5,  # Normalize to approximately -2 to 2
        '1': (usage_hours - 12.0) / 6.0,  # Normalize
        '2': (energy_consumption - 5.0) / 2.5,  # Normalize
        '3': (user_prefs - 0.5) * 2.0,  # Normalize
        '4': (malfunction_incidents - 2.0) / 2.0,  # Normalize
        '5': (device_age - 30.0) / 15.0,  # Normalize
        '6': float(device_encoding[0]),  # Camera
        '7': float(device_encoding[1]),  # Lights
        '8': float(device_encoding[2]),  # Security System
        '9': float(device_encoding[3]),  # Smart Speaker
        '10': float(device_encoding[4])  # Thermostat
    }

def make_prediction(input_data, model):
    """Make prediction using the model"""
    # Convert input to DataFrame (features are already normalized/processed)
    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

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
    """Create comprehensive model performance dashboard with detailed comparative analysis"""
    st.markdown("### 📊 Model Performance Dashboard - Comprehensive Analysis")

    # ============ KEY METRICS SECTION ============
    st.markdown("#### 🎯 Key Performance Indicators")
    best_model = model_comparison['f1'].idxmax()
    best_f1 = model_comparison['f1'].max()
    best_accuracy = model_comparison['accuracy'].idxmax()
    best_roc = model_comparison['roc_auc'].idxmax()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Best F1 Model", best_model, f"{best_f1:.4f}")
    with col2:
        st.metric("Best Accuracy", best_accuracy, f"{model_comparison['accuracy'].max():.4f}")
    with col3:
        st.metric("Best ROC AUC", best_roc, f"{model_comparison['roc_auc'].max():.4f}")
    with col4:
        st.metric("Total Models", len(model_comparison))
    with col5:
        avg_f1 = model_comparison['f1'].mean()
        st.metric("Avg F1-Score", f"{avg_f1:.4f}")

    st.divider()

    # ============ METRIC STATISTICS ============
    st.markdown("#### 📈 Metric Statistics")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    stats_data = []
    
    for metric in metrics:
        stats_data.append({
            'Metric': metric.title(),
            'Min': f"{model_comparison[metric].min():.4f}",
            'Max': f"{model_comparison[metric].max():.4f}",
            'Mean': f"{model_comparison[metric].mean():.4f}",
            'Std Dev': f"{model_comparison[metric].std():.4f}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.divider()

    # ============ BEST MODEL PER METRIC ============
    st.markdown("#### 🏆 Best Model Per Metric")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Classification Metrics")
        best_metrics = {
            '🎯 Accuracy': model_comparison['accuracy'].idxmax(),
            '✅ Precision': model_comparison['precision'].idxmax(),
            '🔍 Recall': model_comparison['recall'].idxmax(),
            '⚖️ F1-Score': model_comparison['f1'].idxmax(),
        }
        for metric, model in best_metrics.items():
            st.write(f"**{metric}**: {model}")
    
    with col2:
        st.subheader("ROC Metric")
        st.write(f"**🎪 ROC AUC**: {model_comparison['roc_auc'].idxmax()}")
    
    with col3:
        st.subheader("Performance Range")
        st.write(f"**Accuracy Range**: {model_comparison['accuracy'].min():.3f} - {model_comparison['accuracy'].max():.3f}")
        st.write(f"**Precision Range**: {model_comparison['precision'].min():.3f} - {model_comparison['precision'].max():.3f}")
        st.write(f"**Recall Range**: {model_comparison['recall'].min():.3f} - {model_comparison['recall'].max():.3f}")

    st.divider()

    # ============ COMPREHENSIVE COMPARISON CHARTS ============
    st.markdown("#### 📊 Visual Comparisons")

    # Performance comparison chart - grouped bar
    fig = px.bar(model_comparison.reset_index(),
                 x='index', y=['accuracy', 'precision', 'recall', 'f1'],
                 title="Classification Metrics Comparison",
                 barmode='group',
                 labels={'index': 'Model', 'value': 'Score'},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=450, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # ROC AUC and Accuracy side-by-side
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_roc = px.bar(model_comparison.reset_index(),
                        x='index', y='roc_auc',
                        title="ROC AUC Comparison",
                        color='roc_auc',
                        color_continuous_scale='Viridis',
                        labels={'index': 'Model', 'roc_auc': 'ROC AUC'})
        fig_roc.update_layout(height=400)
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with col_chart2:
        fig_acc = px.bar(model_comparison.reset_index(),
                        x='index', y='accuracy',
                        title="Accuracy Comparison",
                        color='accuracy',
                        color_continuous_scale='Blues',
                        labels={'index': 'Model', 'accuracy': 'Accuracy'})
        fig_acc.update_layout(height=400)
        st.plotly_chart(fig_acc, use_container_width=True)

    st.divider()

    # ============ HEATMAP FOR DETAILED ANALYSIS ============
    st.markdown("#### 🔥 Performance Heatmap")
    
    # Normalize data for heatmap (0-1)
    heatmap_data = model_comparison[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].copy()
    heatmap_data = heatmap_data.reset_index()
    heatmap_data.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=model_comparison[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].values,
        x=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
        y=model_comparison.index,
        colorscale='RdYlGn',
        text=model_comparison[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig_heatmap.update_layout(height=450, title="All Metrics Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # ============ MODEL RANKINGS ============
    st.markdown("#### 🥇 Model Rankings")
    
    ranking_tabs = st.tabs(["F1-Score", "Accuracy", "ROC AUC", "Overall"])
    
    with ranking_tabs[0]:
        f1_ranked = model_comparison[['f1']].sort_values('f1', ascending=False).reset_index()
        f1_ranked['Rank'] = range(1, len(f1_ranked) + 1)
        f1_ranked.columns = ['Model', 'F1-Score', 'Rank']
        f1_ranked = f1_ranked[['Rank', 'Model', 'F1-Score']]
        st.dataframe(f1_ranked, use_container_width=True, hide_index=True)
    
    with ranking_tabs[1]:
        acc_ranked = model_comparison[['accuracy']].sort_values('accuracy', ascending=False).reset_index()
        acc_ranked['Rank'] = range(1, len(acc_ranked) + 1)
        acc_ranked.columns = ['Model', 'Accuracy', 'Rank']
        acc_ranked = acc_ranked[['Rank', 'Model', 'Accuracy']]
        st.dataframe(acc_ranked, use_container_width=True, hide_index=True)
    
    with ranking_tabs[2]:
        roc_ranked = model_comparison[['roc_auc']].sort_values('roc_auc', ascending=False).reset_index()
        roc_ranked['Rank'] = range(1, len(roc_ranked) + 1)
        roc_ranked.columns = ['Model', 'ROC AUC', 'Rank']
        roc_ranked = roc_ranked[['Rank', 'Model', 'ROC AUC']]
        st.dataframe(roc_ranked, use_container_width=True, hide_index=True)
    
    with ranking_tabs[3]:
        # Calculate overall score (average of all normalized metrics)
        overall_score = (
            (model_comparison['accuracy'] / model_comparison['accuracy'].max()) * 0.25 +
            (model_comparison['precision'] / model_comparison['precision'].max()) * 0.25 +
            (model_comparison['recall'] / model_comparison['recall'].max()) * 0.25 +
            (model_comparison['roc_auc'] / model_comparison['roc_auc'].max()) * 0.25
        ).sort_values(ascending=False)
        
        overall_ranked = pd.DataFrame({
            'Rank': range(1, len(overall_score) + 1),
            'Model': overall_score.index,
            'Overall Score': overall_score.values
        })
        st.dataframe(overall_ranked, use_container_width=True, hide_index=True)

    st.divider()

    # ============ DETAILED PERFORMANCE TABLE ============
    st.markdown("#### 📋 Complete Performance Table")
    
    detailed_table = model_comparison[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].copy()
    detailed_table.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    detailed_table = detailed_table.round(4)
    
    st.dataframe(detailed_table, use_container_width=True)

    st.divider()

    # ============ MODEL COMPARISON INSIGHTS ============
    st.markdown("#### 💡 Key Insights")
    
    col_insight1, col_insight2, col_insight3 = st.columns(3)
    
    with col_insight1:
        st.info(f"""
        **Top Performer (F1)**
        
        Model: {best_model}
        F1-Score: {best_f1:.4f}
        """)
    
    with col_insight2:
        most_consistent = model_comparison[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].std(axis=1).idxmin()
        st.success(f"""
        **Most Consistent**
        
        Model: {most_consistent}
        Std Dev: {model_comparison[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].std(axis=1).min():.4f}
        """)
    
    with col_insight3:
        avg_accuracy = model_comparison['accuracy'].mean()
        st.warning(f"""
        **Average Performance**
        
        Mean Accuracy: {avg_accuracy:.4f}
        Mean F1-Score: {model_comparison['f1'].mean():.4f}
        """)

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

def batch_prediction_interface(models):
    """Batch prediction interface"""
    st.markdown("### 📤 Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file with device data (must have features 0-10)", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())

            # Check if required features exist
            required_features = [str(i) for i in range(11)]
            if not all(feature in batch_data.columns for feature in required_features):
                st.error(f"CSV must contain columns: {', '.join(required_features)}")
                return

            if st.button("Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    # Make predictions
                    dt_model = models['Decision Tree']
                    # Use only the required features
                    batch_features = batch_data[required_features].copy()
                    predictions = dt_model.predict(batch_features)
                    probabilities = dt_model.predict_proba(batch_features)

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

    # Create more readable output with feature names
    feature_names = get_feature_names()
    result_data = {
        feature_names[i]: input_data[str(i)] for i in range(len(feature_names))
    }
    result_data['Predicted_Efficiency'] = prediction
    result_data['Efficiency_Status'] = efficiency_status
    result_data['Confidence_Score'] = f"{confidence:.1f}%"

    result_df = pd.DataFrame([result_data])
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="single_prediction.csv">Download Prediction CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Also show interpretation
    st.markdown("### 📋 Prediction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Efficiency Status", efficiency_status)
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
    with col3:
        st.metric("Prediction Code", int(prediction))

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Smart Home Device Efficiency Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict whether a smart home device is efficient or inefficient using machine learning models.")

    # Load models
    models, model_comparison = load_models_and_preprocessor()

    if models is None:
        st.error("Failed to load models. Please ensure the required files are present.")
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
                prediction, prediction_proba = make_prediction(input_data, dt_model)

                if prediction is not None:
                    efficiency_status, confidence = display_prediction(prediction, prediction_proba)

                    # Export option
                    export_single_prediction(input_data, prediction, confidence, efficiency_status)

    elif page == "Model Dashboard":
        create_model_dashboard(model_comparison)

    elif page == "Feature Importance":
        create_feature_importance_plot(models)

    elif page == "Batch Prediction":
        batch_prediction_interface(models)

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit | Using Decision Tree Model for Predictions")

if __name__ == "__main__":
    main()