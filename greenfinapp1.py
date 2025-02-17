import streamlit as st
st.set_page_config(
    page_title="ESG Project Predictor",
    page_icon="🌍",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

# Load models with diagnostic printing
@st.cache_resource
def load_models():
    with open(r'D:\User1\GREENFIN\greenfinproject_models.pkl', 'rb') as f:
        models = pickle.load(f)
    with open(r'D:\User1\GREENFIN\greenfinscalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    with open(r'D:\User1\GREENFIN\greenfinfeature_mappings.pkl', 'rb') as f:
        feature_mappings = pickle.load(f)
    return models, scalers, feature_mappings

models, scalers, feature_mappings = load_models()

def predict_project_outcomes(data, selected_projects, prediction_years, country, total_budget):
    melted_df = pd.melt(
        data,
        id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
        var_name='Year',
        value_name='Value'
    )
    
    melted_df['Year'] = melted_df['Year'].str.extract(r'(\d{4})').astype(int)
    last_year = melted_df['Year'].max()
    future_years = pd.DataFrame({'Year': range(last_year + 1, last_year + prediction_years + 1)})
    
    reshaped_data = melted_df.pivot_table(
        index=['Country Name', 'Country Code', 'Year'],
        columns='Series Code',
        values='Value'
    ).reset_index()
    
    predictions = {}
    total_cost = 0
    
    for project in selected_projects:
        if project not in models:
            st.warning(f"Project '{project}' not found in trained models")
            continue
            
        features = feature_mappings[project]
        available_features = [f for f in features if f in reshaped_data.columns]
        
        if not available_features:
            st.warning(f"No features available for {project}")
            continue
        
        last_values = reshaped_data[available_features].iloc[-1].values
        future_features = np.array([last_values * (1 + np.random.uniform(-0.05, 0.05, len(last_values))) 
                                  for _ in range(prediction_years)])
        
        scaled_data = scalers[project].transform(future_features)
        project_models = models[project]
        
        project_cost = project_models['cost'].predict(scaled_data).mean()
        
        if total_cost + project_cost <= total_budget:
            total_cost += project_cost
            
            predictions[project] = {
                'esg_scores': project_models['esg'].predict(scaled_data) * (1 + np.random.uniform(-0.02, 0.02, prediction_years)),
                'risk_factors': project_models['risk'].predict(scaled_data) * (1 + np.random.uniform(-0.03, 0.03, prediction_years)),
                'costs': np.minimum(project_models['cost'].predict(scaled_data), total_budget - total_cost),
                'years': future_years['Year'].values
            }
        else:
            st.warning(f"Project {project} exceeds remaining budget of ${total_budget - total_cost:,.2f}")
    
    return predictions

def create_budget_chart(predictions, total_budget):
    project_costs = {project: pred['costs'].mean() for project, pred in predictions.items()}
    total_allocated = sum(project_costs.values())
    remaining = max(0, total_budget - total_allocated)
    
    values = list(project_costs.values()) + [remaining]
    labels = list(project_costs.keys()) + ['Emergency Fund']
    
    fig = px.pie(
        values=values,
        names=labels,
        title='Budget Allocation',
        hole=0.3
    )
    return fig

def create_performance_chart(predictions, selected_projects):
    fig = go.Figure()
    
    for project in selected_projects:
        if project in predictions:
            fig.add_trace(go.Bar(
                name=f'{project} - ESG Score',
                y=predictions[project]['esg_scores'],
                x=predictions[project]['years'],
                text=np.round(predictions[project]['esg_scores'], 2),
                textposition='auto',
            ))
            fig.add_trace(go.Bar(
                name=f'{project} - Risk Factor',
                y=predictions[project]['risk_factors'],
                x=predictions[project]['years'],
                text=np.round(predictions[project]['risk_factors'], 2),
                textposition='auto',
            ))
    
    fig.update_layout(
        title='Project Performance Metrics Over Time',
        barmode='group',
        xaxis_title='Year',
        yaxis_title='Score/Risk Percentage'
    )
    return fig

def main():
    st.title("ESG Investment Project Predictor")
    
    # Display available projects
    st.sidebar.header("Available Projects")
    available_projects = list(models.keys())
    
    st.sidebar.header("Upload Data and Set Parameters")
    uploaded_file = st.sidebar.file_uploader("Upload your ESG dataset (CSV)", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        
        total_budget = st.sidebar.number_input(
            "Total Investment Budget ($)", 
            min_value=0, 
            value=10000000,
            format="%d"
        )
        
        selected_projects = st.sidebar.multiselect(
            "Select Projects",
            options=available_projects
        )
        
        country = st.sidebar.text_input("Enter Country Name")
        prediction_years = st.sidebar.number_input("Years to Predict", min_value=1, value=5)
        
        if st.sidebar.button("Generate Predictions"):
            if selected_projects and country:
                predictions = predict_project_outcomes(
                    data, 
                    selected_projects, 
                    prediction_years,
                    country,
                    total_budget
                )
                
                if predictions:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_budget_chart(predictions, total_budget))
                    with col2:
                        st.plotly_chart(create_performance_chart(predictions, selected_projects))
                    
                    st.subheader("Detailed Project Predictions")
                    for project in predictions:
                        st.write(f"\n{project}")
                        project_df = pd.DataFrame({
                            'Year': predictions[project]['years'],
                            'ESG Score': np.round(predictions[project]['esg_scores'], 2),
                            'Risk Factor (%)': np.round(predictions[project]['risk_factors'], 2),
                            'Estimated Cost ($)': np.round(predictions[project]['costs'], 2)
                        })
                        st.write(project_df)
                        
                        avg_cost = predictions[project]['costs'].mean()
                        st.info(f"Average Project Cost: ${avg_cost:,.2f}")
            else:
                st.warning("Please select at least one project and enter a country name")

if __name__ == "__main__":
    main()
