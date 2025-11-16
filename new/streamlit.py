import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Schneider Electric: Market Strategy Explainer",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GROUNDING PROMPT
# ============================================================================

GROUNDING_PROMPT = """
You are a helpful sales analyst for Schneider Electric. Your job is to answer questions from a non-technical sales manager about why our model predicts some opportunities as 'Won' or 'Lost'.

You MUST answer all questions using **only** the following set of "Model Facts".

**CRITICAL RULES:**
* **Do not use technical jargon** like 'SHAP', 'model', or 'feature'. Use simple business terms. Instead of 'cust_hitrate', say 'customer's past conversion rate' or 'how often this customer buys from us'.
* **Do not make up information.** If the answer is not in the facts below, state that the analysis doesn't have that specific detail.
* **Do not assign the products to some type.** Always refer to them as **products or deals**.
* Be concise and direct.
* When explaining predictions, focus on the business impact and actionable insights.

---

**MODEL FACTS:**

1.  **Top Factor: Customer Conversion Rate (`cust_hitrate`)**
    * This is the **single most important factor** in our predictions by far.
    * A **high** conversion rate (customer frequently buys from us) very strongly predicts a **'Won'** deal.
    * A **low** conversion rate (customer rarely converts) very strongly predicts a **'Lost'** deal.
    * This factor has roughly **twice the impact** of any other factor.

2.  **Second Factor: Opportunity Age (`opp_old`)**
    * This is the second most important factor.
    * If an opportunity has been open for a **long time** (old/stale deal), it is a **very strong predictor** that the deal will be **'Lost'**.
    * Fresh, newer opportunities are more likely to close successfully.
    * **Action insight**: Aging deals should be urgently addressed or disqualified.

3.  **Third Factor: Customer Interactions (`cust_interactions`)**
    * This is the third most important factor, but the relationship is **complex**.
    * The **number of interactions matters**, but it's not simply "more is better."
    * Both very high and very low interaction counts can push predictions in either direction.
    * The impact depends on other factors like customer history and deal context.

4.  **Important Factor: Opportunity Timing (`opp_month`)**
    * The **time of year** when the opportunity was created matters significantly.
    * There are **strong seasonal patterns** in deal success.
    * Some months show higher win rates than others, creating wide variation in predictions.

5.  **Moderate Factors: Customer Characteristics**
    * **Number of contracts** (`cust_contracts`) with the customer has a moderate impact.
    * Some contract configurations strongly favor winning, while others strongly favor losing.
    * Being a customer **in Iberia/Spain** (`cust_in_iberia`) is a **moderate positive sign** (predicts 'Won').

6.  **Product History - Product B**
    * Whether we've sold **Product B** to this customer before (`product_B_sold_in_the_past`) has a **moderate mixed impact**.
    * The effect varies - sometimes it's positive, sometimes negative, depending on context.

7.  **Product History - Product A**
    * Having **recommended Product A** before (`product_A_recommended`) shows balanced impact around neutral.
    * Having **sold Product A** before (`product_A_sold_in_the_past`) shows balanced impact around neutral.
    * The current **amount of Product A** in the deal has balanced impact around neutral.
    * These are contextual modifiers rather than strong predictors.

8.  **Competitors**
    * **Competitor Z** (`competitor_Z`) has a **moderate negative impact** (predicts 'Lost').
    * **Competitor Y** (`competitor_Y`) has a **moderate negative impact** (predicts 'Lost').
    * **Competitor X** (`competitor_X`) has **minimal to no impact** on predictions.
    * Overall, competitor presence matters **less than customer characteristics**.

9.  **Low-Impact Factors**
    * **Product C** (`product_C`) in the current deal has **very minimal impact**.
    * **Product D** (`product_D`) in the current deal has **very minimal impact**.
    * These products don't significantly drive win/loss predictions.

---

**SUMMARY - What Matters Most:**
The model is primarily driven by **customer behavior patterns** (conversion history, interaction patterns) and **deal characteristics** (age, timing), rather than by product mix or competitive factors. Focus on customers with strong track records and address aging opportunities quickly.
"""

# ============================================================================
# API CONFIGURATION
# ============================================================================

@st.cache_resource
def configure_gemini():
    """Configure Gemini API with proper error handling."""
    try:
        # Try Streamlit secrets first, then environment variable
        api_key =os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            st.error("⚠️ **API Key Missing**: Please configure GOOGLE_API_KEY in Streamlit secrets or environment variables.")
            st.stop()
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {e}")
        st.stop()

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model_and_explainer():
    """Load the ML model and SHAP explainer with error handling."""
    model_path = Path('./model/random_forest_model.pkl')
    
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path.absolute()}")
        st.info("Please ensure `random_forest_model.pkl` is in the `./model/` directory.")
        st.stop()
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        explainer = shap.TreeExplainer(model)
        
        # Validate model has required methods
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            st.error("❌ Loaded model doesn't have required prediction methods.")
            st.stop()
        
        return model, explainer
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ============================================================================
# LLM EXPLANATION HELPERS
# ============================================================================

def get_gemini_local_explanation(gemini_model, prediction_text, local_shap_factors):
    """
    Generate plain-English explanation for a specific prediction.
    
    Args:
        gemini_model: Configured Gemini model instance
        prediction_text: "WON" or "LOST"
        local_shap_factors: Dict of {feature_name: shap_value}
    
    Returns:
        str: Natural language explanation
    """
    factors_list = []
    for feature, value in local_shap_factors.items():
        impact_strength = "strong" if abs(value) > 0.1 else "slight"
        impact_direction = "positive" if value > 0 else "negative"
        factors_list.append(f"- '{feature}': had a {impact_strength} {impact_direction} impact (value: {value:.3f})")
    
    factors_string = "\n".join(factors_list)

    prompt = f"""
You are a sales analyst explaining *one specific* prediction to a sales manager.
The model predicted this opportunity will be: {prediction_text}

These are the top factors that influenced the decision:
{factors_string}

Please write a 2-3 sentence, user-friendly summary explaining *why* the model made this prediction.
Do not use technical terms like 'SHAP', 'model', or 'feature'. Use business language.
Focus on actionable insights.
"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"

# ============================================================================
# FEATURE INPUT CONFIGURATION
# ============================================================================

FEATURE_CONFIG = {
    'cust_hitrate': {
        'label': 'Customer Hit Rate',
        'widget': 'slider',
        'min': -1.16670,
        'max': 1.73092,
        'default': 0.12643,
        'help': 'Customer conversion rate (normalized)'
    },
    'cust_interactions': {
        'label': 'Customer Interactions',
        'widget': 'slider',
        'min': -0.68170,
        'max': 13.22972,
        'default': 0.30731,
        'help': 'Number of touchpoints with this customer'
    },
    'opp_old': {
        'label': 'Opportunity is Old?',
        'widget': 'slider',
        'min': -0.28185,
        'max': 3.54793,
        'default': -0.28185,
        'help': 'Has this opportunity been open for a long time?'
    },
    'opp_month': {
        'label': 'Opportunity Month',
        'widget': 'slider',
        'min': -1.41464,
        'max': 1.83486,
        'default': -1.41464,
        'help': 'Month when opportunity was created (normalized)'
    },
    'cust_contracts': {
        'label': 'Customer Contracts',
        'widget': 'slider',
        'min': -0.34997,
        'max': 10.46037,
        'default': -0.34997,
        'help': 'Number of active contracts with this customer'
    },
    'cust_in_iberia': {
        'label': 'Customer in Iberia?',
        'widget': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'Yes' if x == 1 else 'No',
        'default': 1,
        'help': 'Is the customer located in Spain/Portugal?'
    },
    'product_A': {
        'label': 'Product A (Current Deal)',
        'widget': 'slider',
        'min': -0.08910,
        'max': 18.78559,
        'default': -0.08910,
        'help': 'Amount of Product A in this opportunity (normalized)'
    },
    'product_A_sold_in_the_past': {
        'label': 'Product A (Past Sales)',
        'widget': 'slider',
        'min': -0.25992,
        'max': 7.95386,
        'default': -0.25992,
        'help': 'Historical Product A sales volume (normalized)'
    },
    'product_A_recommended': {
        'label': 'Product A Recommended Before?',
        'widget': 'slider',
        'min': -0.1097,
        'max': 23.323,
        'default': -0.1097,
    },
    'product_B_sold_in_the_past': {
        'label': 'Product B (Past Sales)',
        'widget': 'slider',
        'min': -0.34794,
        'max': 8.54676,
        'default': -0.34794,
        'help': 'Historical Product B sales volume (normalized)'
    },
    'product_C': {
        'label': 'Product C (Current Deal)',
        'widget': 'slider',
        'min': -0.02372,
        'max': 65.04750,
        'default': -0.02372
    },
    'product_D': {
        'label': 'Product D (Current Deal)',
        'widget': 'slider',
        'min': -0.04247,
        'max': 36.23157,
        'default': -0.04247
    },
    'competitor_X': {
        'label': 'Competitor X Present?',
        'widget': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'Yes' if x == 1 else 'No',
        'default': 0
    },
    'competitor_Y': {
        'label': 'Competitor Y Present?',
        'widget': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'Yes' if x == 1 else 'No',
        'default': 0
    },
    'competitor_Z': {
        'label': 'Competitor Z Present?',
        'widget': 'selectbox',
        'options': [0, 1],
        'format_func': lambda x: 'Yes' if x == 1 else 'No',
        'default': 0
    }
}

def render_feature_inputs():
    """Render all feature inputs in sidebar and return collected values."""
    st.sidebar.header("📝 Enter Opportunity Data")
    
    values = {}
    
    # Group features by category
    st.sidebar.subheader("Customer Metrics")
    for feature in ['cust_hitrate', 'cust_interactions', 'cust_contracts', 'cust_in_iberia']:
        values[feature] = render_single_input(feature, FEATURE_CONFIG[feature])
    
    st.sidebar.subheader("Opportunity Details")
    for feature in ['opp_old', 'opp_month']:
        values[feature] = render_single_input(feature, FEATURE_CONFIG[feature])
    
    st.sidebar.subheader("Product History")
    for feature in ['product_A', 'product_A_sold_in_the_past', 'product_A_recommended',
                    'product_B_sold_in_the_past', 'product_C', 'product_D']:
        values[feature] = render_single_input(feature, FEATURE_CONFIG[feature])
    
    st.sidebar.subheader("Competitive Landscape")
    for feature in ['competitor_X', 'competitor_Y', 'competitor_Z']:
        values[feature] = render_single_input(feature, FEATURE_CONFIG[feature])
    
    return values

def render_single_input(feature_name, config):
    """Render a single input widget based on configuration."""
    widget = config['widget']
    label = config['label']
    
    if widget == 'slider':
        value = st.sidebar.slider(
            label,
            min_value=float(config['min']),
            max_value=float(config['max']),
            value=float(config['default']),
            help=config.get('help'),
            step=0.01
        )
        return value
    
    elif widget == 'number':
        return st.sidebar.number_input(
            label,
            min_value=config['min'],
            value=config['default'],
            help=config.get('help')
        )
    
    elif widget == 'selectbox':
        return st.sidebar.selectbox(
            label,
            options=config['options'],
            index=config['options'].index(config['default']),
            format_func=config.get('format_func'),
            help=config.get('help')
        )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Initialize resources
    model, explainer = load_model_and_explainer()
    gemini_model = configure_gemini()
    
    # Header
    st.title("Schneider Electric: Market Strategy Explainer")
    st.markdown("**Understand** why our model predicts deals as Won or Lost, and **explore** live predictions.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Ask About the Model", "🔮 Live Prediction"])
    
    # ========================================================================
    # TAB 1: GLOBAL CHATBOT
    # ========================================================================
    with tab1:
        st.header("Ask About Our Model's General Strategy")
        st.markdown(
            "Ask questions like:\n"
            "- *Which competitors matter most?*\n"
            "- *What's the most important factor for winning?*\n"
            "- *How does customer history affect predictions?*"
        )
        
        # Initialize chat history
        if "chat" not in st.session_state:
            st.session_state.chat = gemini_model.start_chat(history=[
                {'role': 'user', 'parts': [GROUNDING_PROMPT]},
                {'role': 'model', 'parts': ["Understood. I am a sales analyst for Schneider Electric. I am ready to answer questions about our model's behavior based on the facts provided."]}
            ])
        
        # Display chat history (skip the grounding prompt)
        for message in st.session_state.chat.history[2:]:
            with st.chat_message(message.role):
                st.markdown(message.parts[0].text)
        
        # Chat input
        if prompt := st.chat_input("What would you like to know about the model?"):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat.send_message(prompt)
                    if response.parts:
                        with st.chat_message("assistant"):
                            st.markdown(response.text)
                    else:
                        # This means the response was empty (likely blocked)
                        st.error("The model's response was blocked. This can happen due to safety filters. Try rephrasing your question.")
                        # Optional: Log the full response to see the safety ratings
                        print("BLOCKED RESPONSE:", response)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ========================================================================
    # TAB 2: LIVE PREDICTION
    # ========================================================================
    with tab2:
        st.header("Get a Live Prediction for a New Opportunity")
        st.markdown("Use the sidebar to configure opportunity details, then click **Predict** to see the outcome and explanation.")
        
        # Render inputs
        input_values = render_feature_inputs()
        
        # Prediction button
        if st.sidebar.button("🎯 Predict and Explain", type="primary", use_container_width=True):
            
            # Create DataFrame with correct feature order
            new_data_df = pd.DataFrame([input_values])
            
            # Ensure column order matches model training
            if hasattr(model, 'feature_names_in_'):
                new_data_df = new_data_df[model.feature_names_in_]
            
            # Display input summary
            with st.expander("📋 Input Data Summary", expanded=False):
                st.dataframe(new_data_df, use_container_width=True)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                try:
                    prediction = model.predict(new_data_df)[0]
                    prediction_proba = model.predict_proba(new_data_df)[0]
                    
                    prediction_text = "WON" if prediction == 1 else "LOST"
                    probability_percent = prediction_proba[prediction] * 100
                    
                    # Display prediction
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(
                            label="Prediction",
                            value=prediction_text,
                            delta=f"{probability_percent:.1f}% confident"
                        )
                    
                    # Calculate SHAP values
                    with st.spinner("Generating explanation..."):
                        shap_values_array = explainer.shap_values(new_data_df)
                        
                        # Extract SHAP values for positive class
                        # For binary classification: shap_values_array is list of 2 arrays
                        if isinstance(shap_values_array, list):
                            shap_values_won = shap_values_array[1][0]  # Class 1, first sample
                        else:
                            shap_values_won = shap_values_array[0,:,1]  # Single array case
                        
                        # Generate waterfall plot
                        st.subheader("📊 Why did the model decide this?")
                        st.markdown("Factors pushing toward **Won** (red/pink) vs **Lost** (blue):")
                        
                        # Get base value (expected value for positive class)
                        if isinstance(explainer.expected_value, (list, np.ndarray)):
                            if len(explainer.expected_value) > 1:
                                base_value = float(explainer.expected_value[1])  # Class 1
                            else:
                                base_value = float(explainer.expected_value[0])
                        else:
                            base_value = float(explainer.expected_value)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values_won,
                                base_values=base_value,
                                data=new_data_df.iloc[0].values,
                                feature_names=new_data_df.columns.tolist()
                            ),
                            show=False
                        )
                        st.pyplot(fig, bbox_inches='tight', use_container_width=True)
                        plt.close()
                        
                        # Generate LLM explanation
                        st.subheader("🤖 Plain-English Summary")
                        
                        # Get top 5 factors
                        feature_names = new_data_df.columns.tolist()
                        top_indices = np.argsort(np.abs(shap_values_won))[-5:]
                        top_factors = {
                            feature_names[i]: float(shap_values_won[i])
                            for i in reversed(top_indices)
                        }
                        
                        explanation = get_gemini_local_explanation(
                            gemini_model,
                            prediction_text,
                            top_factors
                        )
                        st.info(explanation)
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.exception(e)

if __name__ == "__main__":
    main()