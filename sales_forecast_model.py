import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesForecastModel:
    def __init__(self):
        """Initialize the SalesForecastModel with default parameters"""
        self.model = None
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.features = None
        self.fitted_model = None
        self.train_features = None
        self.test_features = None
        self.feature_names = None
        
    def _clean_and_validate_data(self, df):
        """
        Clean and validate the data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe to clean
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned dataframe
        """
        # Check for required columns
        required_columns = ['Sales', 'Marketing_Spend', 'Economic_Index', 
                          'Competitor_Price_Index', 'Sales_t-1', 'Sales_t-2']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Handle missing values
        if df.isnull().any().any():
            logger.info("Found missing values. Applying forward fill followed by backward fill...")
            df = df.fillna(method='ffill').fillna(method='bfill')
            
        # Handle infinite values
        if np.isinf(df.values).any():
            logger.info("Found infinite values. Replacing with NaN and then filling...")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
        # Validate numeric columns
        for col in required_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Column {col} must contain numeric values")
                
        return df
        
    def _parse_dates(self, date_series):
        """
        Try different date parsing approaches
        
        Parameters:
        -----------
        date_series : pandas.Series
            Series containing date strings
            
        Returns:
        --------
        pandas.Series
            Series with parsed dates
        """
        # Try different date formats
        date_formats = [
            'mixed',  # Let pandas infer the format for each date
            '%Y-%m-%d',  # ISO format
            '%m/%d/%Y',  # US format
            '%d/%m/%Y',  # UK format
            '%Y/%m/%d',  # Alternative ISO format
            '%d-%m-%Y',  # European format
            '%m-%d-%Y'   # US format with dashes
        ]
        
        for date_format in date_formats:
            try:
                if date_format == 'mixed':
                    return pd.to_datetime(date_series, format='mixed')
                else:
                    return pd.to_datetime(date_series, format=date_format)
            except (ValueError, TypeError):
                continue
                
        # If none of the formats work, try the most flexible parsing
        try:
            return pd.to_datetime(date_series, infer_datetime_format=True)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not parse dates. Please ensure dates are in a consistent format. Error: {str(e)}")
        
    def load_data(self, file_path):
        """
        Load and prepare the sales data
        """
        try:
            # Determine file type and read accordingly
            if isinstance(file_path, str):
                file_ext = os.path.splitext(file_path)[1].lower()
            else:
                # Handle Streamlit's UploadedFile object
                file_ext = os.path.splitext(file_path.name)[1].lower()
            
            logger.info(f"Reading file with extension: {file_ext}")
            
            try:
                if file_ext == '.csv':
                    self.data = pd.read_csv(file_path)
                elif file_ext in ['.xlsx', '.xls']:
                    self.data = pd.read_excel(file_path, engine='openpyxl')
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}. Please use CSV or Excel files.")
            except UnicodeDecodeError:
                # Try reading with different encoding if UTF-8 fails
                if file_ext == '.csv':
                    self.data = pd.read_csv(file_path, encoding='latin1')
                else:
                    raise
            
            # Verify the data is not empty
            if self.data.empty:
                raise ValueError("The uploaded file contains no data")
            
            # Check if Date column exists
            if 'Date' not in self.data.columns:
                raise ValueError("The file must contain a 'Date' column")
            
            # Convert date column - try to handle various date formats
            try:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
            except Exception as e:
                logger.warning(f"Error converting dates: {str(e)}. Trying alternative date formats...")
                try:
                    # Try parsing with different formats
                    self.data['Date'] = pd.to_datetime(self.data['Date'], format='mixed')
                except:
                    raise ValueError("Could not parse the Date column. Please ensure dates are in a valid format.")
            
            self.data.set_index('Date', inplace=True)
            
            # Sort index to ensure chronological order
            self.data = self.data.sort_index()
            
            # Define feature names
            self.feature_names = ['Marketing_Spend', 'Economic_Index', 
                                'Competitor_Price_Index', 'Sales_t-1', 'Sales_t-2']
            
            # Check for required columns
            required_columns = ['Sales'] + self.feature_names
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}. The file must contain: {', '.join(required_columns)}")
            
            # Convert all feature columns to numeric, coercing errors to NaN
            for col in required_columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Handle missing values if any
            if self.data[required_columns].isnull().any().any():
                logger.warning("Found missing or non-numeric values. Filling with forward fill method.")
                self.data[required_columns] = self.data[required_columns].fillna(method='ffill').fillna(method='bfill')
            
            # Scale the features
            self.features = self.data[self.feature_names].copy()
            self.features = pd.DataFrame(
                self.scaler.fit_transform(self.features),
                index=self.data.index,
                columns=self.feature_names
            )
            
            logger.info(f"Data loaded successfully! {len(self.data)} records processed.")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def split_data(self, train_size=0.8):
        """
        Split the data into training and testing sets
        """
        if self.data is None:
            raise ValueError("Please load data first using load_data()")
            
        try:
            # Calculate split point
            split_idx = int(len(self.data) * train_size)
            split_date = self.data.index[split_idx]
            
            # Split target variable
            self.train_data = self.data['Sales'][:split_idx]
            self.test_data = self.data['Sales'][split_idx:]
            
            # Split features
            self.train_features = self.features[:split_idx]
            self.test_features = self.features[split_idx:]
            
            logger.info(f"Data split at {split_date}")
            logger.info(f"Training samples: {len(self.train_data)}")
            logger.info(f"Testing samples: {len(self.test_data)}")
            
        except Exception as e:
            logger.error(f"Error during data splitting: {str(e)}")
            raise
    
    def train_model(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Train the SARIMAX model with exogenous variables
        """
        if self.train_data is None:
            raise ValueError("Please split the data first using split_data()")
            
        try:
            logger.info(f"Training SARIMAX model with order {order} and seasonal_order {seasonal_order}")
            
            # Train SARIMAX model with exogenous variables
            self.model = SARIMAX(
                self.train_data,
                exog=self.train_features,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = self.model.fit(disp=False)
            logger.info("Model trained successfully!")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def make_predictions(self, forecast_periods=30):
        """
        Make predictions on test data and future periods
        """
        if self.fitted_model is None:
            raise ValueError("Please train the model first using train_model()")
            
        try:
            logger.info(f"Making predictions for {forecast_periods} periods ahead")
            
            # Get in-sample predictions (training period)
            in_sample_predictions = pd.Series(
                self.fitted_model.get_prediction(
                    start=0,
                    end=len(self.train_data)-1,
                    exog=self.train_features
                ).predicted_mean,
                index=self.train_data.index
            )
            
            # Get test predictions
            test_predictions = pd.Series(
                self.fitted_model.get_prediction(
                    start=len(self.train_data),
                    end=len(self.data)-1,
                    exog=self.test_features
                ).predicted_mean,
                index=self.test_data.index
            )
            
            # Create future dates
            last_date = self.data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=forecast_periods,
                freq='7D'
            )
            
            # Initialize future features DataFrame with the same structure as training features
            future_features = pd.DataFrame(
                index=future_dates,
                columns=self.feature_names,
                dtype=float
            )
            
            # Use last known values for non-lag features
            for col in ['Marketing_Spend', 'Economic_Index', 'Competitor_Price_Index']:
                future_features[col] = self.data[col].iloc[-1]
            
            # Initialize lagged values
            last_sales = self.data['Sales'].iloc[-1]
            second_last_sales = self.data['Sales'].iloc[-2]
            
            # Generate predictions iteratively
            future_predictions = []
            
            for i in range(forecast_periods):
                # Update lag features
                current_features = future_features.iloc[[i]].copy()
                current_features['Sales_t-1'] = last_sales
                current_features['Sales_t-2'] = second_last_sales
                
                # Scale the features
                scaled_features = pd.DataFrame(
                    self.scaler.transform(current_features),
                    columns=self.feature_names
                )
                
                # Make prediction
                pred = self.fitted_model.forecast(steps=1, exog=scaled_features)[0]
                future_predictions.append(pred)
                
                # Update lagged values for next iteration
                second_last_sales = last_sales
                last_sales = pred
            
            future_predictions = pd.Series(future_predictions, index=future_dates)
            
            logger.info("Predictions generated successfully")
            return in_sample_predictions, test_predictions, future_predictions
            
        except Exception as e:
            logger.error(f"Error during predictions: {str(e)}")
            raise
    
    def evaluate_model(self, test_predictions):
        """
        Evaluate the model performance on test data
        """
        try:
            mae = mean_absolute_error(self.test_data, test_predictions)
            rmse = np.sqrt(mean_squared_error(self.test_data, test_predictions))
            mape = np.mean(np.abs((self.test_data - test_predictions) / self.test_data)) * 100
            r2 = r2_score(self.test_data, test_predictions)
            
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
            logger.info("Model evaluation metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.2f}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise