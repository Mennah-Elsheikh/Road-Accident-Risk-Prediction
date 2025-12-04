"""
Streamlit Web Application for Accident Risk Prediction
Interactive interface for predicting accident risk based on road and environmental conditions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Accident Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = Path(__file__).parent / "best_model.joblib"
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical features to match model training
    """
    # Create a copy
    df_encoded = df.copy()
    
    # Define all categorical columns including booleans
    categorical_cols = [
        'road_type', 'lighting', 'weather', 'time_of_day',
        'road_signs_present', 'public_road', 'holiday', 'school_season'
    ]
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=False)
    
    # Ensure all expected columns are present in the correct order
    # Based on the model expecting 24 features total
    expected_columns = [
        # Numerical features
        'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
        
        # String categorical features
        'road_type_highway', 'road_type_rural', 'road_type_urban',
        'lighting_daylight', 'lighting_dim', 'lighting_night',
        'weather_clear', 'weather_foggy', 'weather_rainy',
        'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_morning',
        
        # Boolean categorical features (One-Hot Encoded)
        'road_signs_present_False', 'road_signs_present_True',
        'public_road_False', 'public_road_True',
        'holiday_False', 'holiday_True',
        'school_season_False', 'school_season_True'
    ]
    
    # Add missing columns with 0
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Select only the expected columns in the correct order
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded


def categorize_risk(risk_value: float) -> tuple[str, str]:
    """
    Categorize risk value into human-readable levels
    Returns: (risk_level, color)
    """
    if risk_value < 0.2:
        return "Very Low", "#28a745"
    elif risk_value < 0.35:
        return "Low", "#5cb85c"
    elif risk_value < 0.5:
        return "Moderate", "#ffc107"
    elif risk_value < 0.7:
        return "High", "#fd7e14"
    else:
        return "Very High", "#dc3545"


def create_risk_gauge(risk_value: float):
    """Create a gauge chart for risk visualization"""
    risk_level, color = categorize_risk(risk_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Accident Risk Score", 'font': {'size': 24}},
        delta={'reference': 35, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#e8f5e9'},
                {'range': [20, 35], 'color': '#c8e6c9'},
                {'range': [35, 50], 'color': '#fff9c4'},
                {'range': [50, 70], 'color': '#ffe0b2'},
                {'range': [70, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_feature_importance_chart(features_dict: dict):
    """Create a bar chart showing input features"""
    df = pd.DataFrame(list(features_dict.items()), columns=['Feature', 'Value'])
    
    fig = px.bar(
        df, 
        x='Value', 
        y='Feature', 
        orientation='h',
        title='Input Features Overview',
        color='Value',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Value",
        yaxis_title="Feature"
    )
    
    return fig


def main():
    # Header
    st.title("🚗 Accident Risk Prediction System")
    st.markdown("### Predict accident risk based on road and environmental conditions")
    
    # Load model
    model, error = load_model()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("Please ensure 'best_model.joblib' exists in the project directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for inputs
    st.sidebar.header("📋 Input Parameters")
    st.sidebar.markdown("---")
    
    # Road Characteristics
    st.sidebar.subheader("🛣️ Road Characteristics")
    road_type = st.sidebar.selectbox(
        "Road Type",
        options=["urban", "rural", "highway"],
        help="Type of road"
    )
    
    num_lanes = st.sidebar.slider(
        "Number of Lanes",
        min_value=1,
        max_value=4,
        value=2,
        help="Number of lanes on the road"
    )
    
    curvature = st.sidebar.slider(
        "Road Curvature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Road curvature (0 = straight, 1 = very curved)"
    )
    
    speed_limit = st.sidebar.select_slider(
        "Speed Limit (mph)",
        options=[25, 35, 45, 60, 70],
        value=45,
        help="Posted speed limit"
    )
    
    # Environmental Conditions
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌤️ Environmental Conditions")
    
    lighting = st.sidebar.selectbox(
        "Lighting Conditions",
        options=["daylight", "dim", "night"],
        help="Current lighting conditions"
    )
    
    weather = st.sidebar.selectbox(
        "Weather Conditions",
        options=["clear", "rainy", "foggy"],
        help="Current weather conditions"
    )
    
    time_of_day = st.sidebar.selectbox(
        "Time of Day",
        options=["morning", "afternoon", "evening"],
        help="Current time of day"
    )
    
    # Road Features
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚦 Road Features")
    
    road_signs_present = st.sidebar.checkbox(
        "Road Signs Present",
        value=True,
        help="Are road signs present?"
    )
    
    public_road = st.sidebar.checkbox(
        "Public Road",
        value=True,
        help="Is this a public road?"
    )
    
    # Contextual Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Contextual Information")
    
    holiday = st.sidebar.checkbox(
        "Holiday",
        value=False,
        help="Is it a holiday?"
    )
    
    school_season = st.sidebar.checkbox(
        "School Season",
        value=True,
        help="Is it during school season?"
    )
    
    num_reported_accidents = st.sidebar.number_input(
        "Previously Reported Accidents",
        min_value=0,
        max_value=10,
        value=1,
        help="Number of previously reported accidents at this location"
    )
    
    # Prepare input data
    input_data = {
        'road_type': road_type,
        'num_lanes': num_lanes,
        'curvature': curvature,
        'speed_limit': speed_limit,
        'lighting': lighting,
        'weather': weather,
        'road_signs_present': road_signs_present,
        'public_road': public_road,
        'time_of_day': time_of_day,
        'holiday': holiday,
        'school_season': school_season,
        'num_reported_accidents': num_reported_accidents
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Prediction Results")
        
        # Predict button
        if st.button("🔮 Predict Accident Risk", type="primary", use_container_width=True):
            with st.spinner("Calculating risk..."):
                try:
                    # Convert to DataFrame
                    df_input = pd.DataFrame([input_data])
                    
                    # Encode categorical features
                    df_input_encoded = encode_categorical_features(df_input)
                    
                    # Make prediction
                    prediction = model.predict(df_input_encoded)[0]
                    prediction = np.clip(prediction, 0.0, 1.0)
                    
                    # Store in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['input_data'] = input_data
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    return
        
        # Display results if prediction exists
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            risk_level, color = categorize_risk(prediction)
            
            # Risk gauge
            st.plotly_chart(create_risk_gauge(prediction), use_container_width=True)
            
            # Risk metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Risk Score",
                    value=f"{prediction:.2%}",
                    delta=f"{(prediction - 0.35):.2%} vs avg"
                )
            
            with metric_col2:
                st.metric(
                    label="Risk Level",
                    value=risk_level
                )
            
            with metric_col3:
                safety_score = (1 - prediction) * 100
                st.metric(
                    label="Safety Score",
                    value=f"{safety_score:.1f}/100"
                )
            
            # Risk interpretation
            st.markdown("---")
            st.subheader("📝 Risk Interpretation")
            
            if risk_level == "Very Low":
                st.success("✅ **Very Low Risk**: Conditions are very safe. Minimal accident risk.")
            elif risk_level == "Low":
                st.success("✅ **Low Risk**: Conditions are generally safe. Exercise normal caution.")
            elif risk_level == "Moderate":
                st.warning("⚠️ **Moderate Risk**: Conditions require attention. Drive carefully.")
            elif risk_level == "High":
                st.warning("⚠️ **High Risk**: Conditions are hazardous. Exercise extreme caution.")
            else:
                st.error("🚨 **Very High Risk**: Conditions are very dangerous. Avoid if possible.")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            recommendations = []
            
            if weather in ["rainy", "foggy"]:
                recommendations.append("🌧️ Reduce speed due to poor weather conditions")
            if lighting in ["dim", "night"]:
                recommendations.append("💡 Use headlights and increase following distance")
            if curvature > 0.7:
                recommendations.append("🔄 Navigate curves carefully and reduce speed")
            if not road_signs_present:
                recommendations.append("⚠️ Be extra vigilant due to lack of road signs")
            if num_reported_accidents > 2:
                recommendations.append("📍 High accident area - exercise extreme caution")
            if speed_limit >= 60:
                recommendations.append("🏎️ High speed area - maintain safe following distance")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ No special recommendations. Drive safely!")
    
    with col2:
        st.subheader("📈 Input Summary")
        
        # Display input parameters
        st.json(input_data)
        
        # Download prediction
        if 'prediction' in st.session_state:
            result_data = {
                **input_data,
                'predicted_risk': float(st.session_state['prediction']),
                'risk_level': categorize_risk(st.session_state['prediction'])[0]
            }
            
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json.dumps(result_data, indent=2),
                file_name="accident_risk_prediction.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>🚗 Accident Risk Prediction System | Built with Streamlit & Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
