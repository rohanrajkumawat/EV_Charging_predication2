import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Set Streamlit page config first thing
st.set_page_config(
    page_title="EV Forecast Dashboard", 
    layout="wide",
    page_icon="🚗"
)

# === Load model ===
@st.cache_resource
def load_model():
    return joblib.load('forecasting_ev_model.pkl')

model = load_model()

# === Styling ===
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #98ff98,#c1fdc1, #d0f0c0);
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #00000 !important;
        }
        .css-1aumxhk {
            background-color: rgba(30, 30, 40, 0.8);
            border-radius: 10px;
            padding: 20px;
        }
        .st-b7 {
            color: white;
        }
        .stSelectbox label, .stMultiselect label {
            color: white !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Header with logo and title
col1, col2 = st.columns([1, 4])
with col1:
    st.image("ev-car-factory.jpg", width=120)
with col2:
    st.markdown("""
        <h1 style='color:light Black; margin-top: 20px;'>
            EV Adoption Forecaster for Washington State
        </h1>
    """, unsafe_allow_html=True)

# Welcome message
st.markdown("""
    <div style='text-align: center; font-size: 18px; padding: 10px; background-color: rgba(255,255,255,1); border-radius: 10px; margin-bottom: 20px;'>
        Predict electric vehicle adoption trends across Washington counties with our advanced forecasting tool
    </div>
""", unsafe_allow_html=True)

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    # Calculate cumulative EV counts
    df['Cumulative EV'] = df.groupby('County')['Electric Vehicle (EV) Total'].cumsum()
    return df

df = load_data()

# Add this once in your app (top or in sidebar)
st.markdown("""
    <style>
    .stSelectbox label {
        color: black !important;
        font-size: 0.9rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
# Sidebar for controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3652/3652191.png", width=80)
    st.markdown("### Dashboard Controls")
    county_list = sorted(df['County'].dropna().unique().tolist())
    selected_county = st.selectbox(" Select Primary Country", county_list, index=county_list.index('King') if 'King' in county_list else 0)
    
    st.markdown("---")
    st.markdown("### Comparison Settings")
    compare_mode = st.checkbox("Enable County Comparison", True)
    if compare_mode:
        compare_counties = st.multiselect(
            "Select up to 3 comparison counties", 
            [c for c in county_list if c != selected_county],
            default=['Pierce', 'Snohomish'] if ('Pierce' in county_list and 'Snohomish' in county_list) else [],
            max_selections=3
        )
    
    st.markdown("---")
    forecast_years = st.slider("Forecast Period (Years)", 1, 5, 3)
    forecast_horizon = forecast_years * 12
    
    st.markdown("---")
    st.markdown("**About This Tool**")
    st.info("""
        This dashboard predicts EV adoption trends using machine learning. 
        The model considers historical trends, seasonality, and county-specific factors.
    """)

# === Forecasting Function ===
def generate_forecast(county, horizon):
    county_df = df[df['County'] == county].sort_values("Date")
    if county_df.empty:
        st.warning(f"No data available for {county}")
        return None
    
    county_code = county_df['county_encoded'].iloc[0]
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()

    future_rows = []
    
    for i in range(1, horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_rows.append({
            "Date": forecast_date, 
            "Predicted EV Total": round(pred),
            "County": county
        })

        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)

        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)

    # Prepare historical data
    historical = county_df[['Date', 'Electric Vehicle (EV) Total', 'Cumulative EV']].copy()
    historical['Type'] = 'Historical'
    historical = historical.rename(columns={'Electric Vehicle (EV) Total': 'EV Count'})
    
    # Prepare forecast data
    forecast = pd.DataFrame(future_rows)
    forecast['Cumulative EV'] = forecast['Predicted EV Total'].cumsum() + historical['Cumulative EV'].iloc[-1]
    forecast['Type'] = 'Forecast'
    forecast = forecast.rename(columns={'Predicted EV Total': 'EV Count'})
    
    # Combine
    combined = pd.concat([
        historical[['Date', 'EV Count', 'Cumulative EV', 'Type']].assign(County=county),
        forecast[['Date', 'EV Count', 'Cumulative EV', 'Type']].assign(County=county)
    ], ignore_index=True)
    
    return combined

# === Main Dashboard ===
tab1, tab2, tab3 = st.tabs(["Primary Forecast", "County Comparison", "Data Insights"])

with tab1:
    st.subheader(f"🚗 EV Adoption Forecast for {selected_county} County")
    
    forecast_data = generate_forecast(selected_county, forecast_horizon)
    
    if forecast_data is not None:
        # KPI Cards
        current_ev = forecast_data[forecast_data['Type'] == 'Historical']['Cumulative EV'].iloc[-1]
        projected_ev = forecast_data['Cumulative EV'].iloc[-1]
        growth_pct = ((projected_ev - current_ev) / current_ev) * 100 if current_ev > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current EVs", f"{int(current_ev):,}")
        col2.metric(f"Projected in {forecast_years} Years", f"{int(projected_ev):,}")
        col3.metric("Growth %", f"{growth_pct:.1f}%", delta_color="inverse" if growth_pct < 0 else "normal")
        
        # Interactive Plotly chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Historical data
        hist_data = forecast_data[forecast_data['Type'] == 'Historical']
        fig.add_trace(
            go.Scatter(
                x=hist_data['Date'],
                y=hist_data['Cumulative EV'],
                name='Historical Cumulative',
                line=dict(color='#00CC96', width=3),
                mode='lines+markers'
            ),
            secondary_y=False
        )
        
        # Forecast data
        forecast_data_only = forecast_data[forecast_data['Type'] == 'Forecast']
        fig.add_trace(
            go.Scatter(
                x=forecast_data_only['Date'],
                y=forecast_data_only['Cumulative EV'],
                name='Forecast Cumulative',
                line=dict(color='#EF553B', width=3, dash='dot'),
                mode='lines+markers'
            ),
            secondary_y=False
        )
        
        # Monthly additions (bar chart)
        fig.add_trace(
            go.Bar(
                x=forecast_data['Date'],
                y=forecast_data['EV Count'],
                name='Monthly EV Additions',
                marker_color='rgba(55, 128, 191, 0.7)',
                opacity=0.6
            ),
            secondary_y=True
        )
        
        # Layout
        fig.update_layout(
            title=f'EV Adoption Trend for {selected_county} County',
            xaxis_title='Date',
            yaxis_title='Cumulative EV Count',
            yaxis2_title='Monthly Additions',
            hovermode='x unified',
            template='plotly_dark',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("View Forecast Data"):
            st.dataframe(
                forecast_data.sort_values('Date', ascending=False).reset_index(drop=True).style.format({
                    'EV Count': '{:,}',
                    'Cumulative EV': '{:,}'
                }),
                use_container_width=True
            )

with tab2:
    if compare_mode and compare_counties:
        st.subheader("📊 County Comparison")
        
        # Generate forecasts for all comparison counties
        all_forecasts = [generate_forecast(selected_county, forecast_horizon)]
        for county in compare_counties:
            county_forecast = generate_forecast(county, forecast_horizon)
            if county_forecast is not None:
                all_forecasts.append(county_forecast)
        
        combined_comparison = pd.concat(all_forecasts)
        
        # Comparison chart
        fig = px.line(
            combined_comparison,
            x='Date',
            y='Cumulative EV',
            color='County',
            line_dash='Type',
            title=f'EV Adoption Comparison ({forecast_years} Year Forecast)',
            template='plotly_dark',
            height=600,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(
            hovermode='x unified',
            xaxis_title='Date',
            yaxis_title='Cumulative EV Count',
            legend_title_text='County & Data Type'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth comparison table
        st.subheader("Growth Rate Comparison")
        
        growth_data = []
        for county in [selected_county] + compare_counties:
            county_data = combined_comparison[combined_comparison['County'] == county]
            if not county_data.empty:
                hist = county_data[county_data['Type'] == 'Historical']
                current = hist['Cumulative EV'].iloc[-1]
                projected = county_data['Cumulative EV'].iloc[-1]
                growth = ((projected - current) / current) * 100 if current > 0 else 0
                
                growth_data.append({
                    'County': county,
                    'Current EVs': int(current),
                    f'Projected ({forecast_years}Y)': int(projected),
                    'Growth %': f"{growth:.1f}%",
                    'Growth Rank': growth
                })
        
        growth_df = pd.DataFrame(growth_data).sort_values('Growth Rank', ascending=False)
        st.dataframe(
            growth_df.drop('Growth Rank', axis=1).style.format({
                'Current EVs': '{:,}',
                f'Projected ({forecast_years}Y)': '{:,}'
            }),
            use_container_width=True
        )
        
    else:
        st.warning("Enable county comparison and select at least one county to compare")

with tab3:
    st.subheader("📈 Data Insights")
    
    # Top counties by current EV adoption
    latest_date = df['Date'].max()
    top_counties = df[df['Date'] == latest_date].nlargest(5, 'Cumulative EV')[['County', 'Cumulative EV']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 5 Counties by Current EV Adoption")
        fig = px.bar(
            top_counties,
            x='County',
            y='Cumulative EV',
            color='County',
            text='Cumulative EV',
            template='plotly_dark'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### EV Adoption Distribution")
        fig = px.pie(
            top_counties,
            names='County',
            values='Cumulative EV',
            hole=0.4,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series of all counties
    st.markdown("### Historical EV Adoption Trends")
    fig = px.line(
        df,
        x='Date',
        y='Cumulative EV',
        color='County',
        template='plotly_dark',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: rgba(255,255,255,1); border-radius: 10px;'>
        <p>Developed for AICTE Internship Cycle 2 by S4F</p>
        <p>Data Source: Washington State Department of Licensing | Last Updated: {}</p>
    </div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)
