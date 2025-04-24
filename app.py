import streamlit as st
import pandas as pd
import numpy as np
from sales_forecast_model import SalesForecastModel
import plotly.graph_objects as go

def create_forecast_plot(model, in_sample_pred, test_pred, future_pred):
    """Create an interactive plot using plotly"""
    fig = go.Figure()
    
    # Add historical data
    # Add test data and predictions
    fig.add_trace(
        go.Scatter(x=model.test_data.index, y=model.test_data,
                  name='Test Data', line=dict(color='green', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=test_pred.index, y=test_pred,
                  name='Test Predictions', line=dict(color='lightgreen', width=2))
    )
    
    # Add future predictions
    if len(future_pred) > 0:
        fig.add_trace(
            go.Scatter(x=future_pred.index, y=future_pred,
                      name='Future Forecast', line=dict(color='red', width=2, dash='dash'))
        )
    
    # Update layout
    fig.update_layout(
        title='Sales Forecast Analysis',
        xaxis_title='Date',
        yaxis_title='Sales',
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def main():
    st.set_page_config(page_title="Sales Forecasting App", layout="wide")
    
    st.title("Sales Forecasting Application")
    st.write("""
    Upload your sales data file to generate sales forecasts.
    """)
    
    # Add file format instructions
    st.info("""
    **Required File Format:**
    - File type: CSV
    - Required columns:
        - Date: Date of the sales record
        - Sales: Sales values
        - Marketing_Spend: Marketing expenditure
        - Economic_Index: Economic indicator
        - Competitor_Price_Index: Competitor pricing index
        - Sales_t-1: Sales from previous period
        - Sales_t-2: Sales from two periods ago
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your sales data (CSV file)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            # Initialize model
            model = SalesForecastModel()
            
            # Load and process data
            with st.spinner('Loading and processing data...'):
                model.load_data(uploaded_file)
            
            # Model parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                train_size = st.slider(
                    "Training Data Size",
                    min_value=0.5,
                    max_value=0.9,
                    value=0.8,
                    step=0.1,
                    help="Proportion of data to use for training"
                )
                
            with col2:
                forecast_periods = st.selectbox(
                    "Forecast Periods",
                    options=[4, 8, 12, 16, 24],
                    index=2,
                    help="Number of weeks to forecast"
                )
                
            with col3:
                seasonal_period = st.selectbox(
                    "Seasonal Period",
                    options=[4, 12, 24, 52],
                    index=1,
                    help="Number of periods in one seasonal cycle (e.g., 52 for yearly, 12 for monthly, 4 for quarterly)"
                )
            
            # Train model button
            if st.button("Generate Forecast"):
                try:
                    with st.spinner('Generating forecast...'):
                        # Split data
                        model.split_data(train_size=train_size)
                        
                        # Train model
                        model.train_model(
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, seasonal_period)
                        )
                        
                        # Make predictions
                        in_sample_pred, test_pred, future_pred = model.make_predictions(
                            forecast_periods=forecast_periods
                        )
                        
                        # Show forecast plot
                        st.subheader("Forecast Visualization")
                        fig = create_forecast_plot(model, in_sample_pred, test_pred, future_pred)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display future predictions
                        if len(future_pred) > 0:
                            st.subheader(f"Future Sales Forecast (Next {len(future_pred)} weeks)")
                            forecast_df = pd.DataFrame({
                                'Date': future_pred.index,
                                'Forecasted Sales': future_pred.values.round(2)
                            })
                            st.dataframe(forecast_df)
                            
                            # Download button for forecast results
                            csv = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="Download Forecast Results",
                                data=csv,
                                file_name="sales_forecast.csv",
                                mime="text/csv"
                            )
                            
                except Exception as e:
                    st.error(f"An error occurred during forecast generation: {str(e)}")
                    
        except Exception as e:
            st.error(f"An error occurred while loading the data: {str(e)}")
            st.write("Please ensure your file matches the required format.")

if __name__ == "__main__":
    main() 