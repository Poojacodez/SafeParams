import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import re
import json

class MLInputValidator:
    """
    A machine learning-based input validation system that learns normal patterns
    in data and detects anomalous inputs.
    """
   
    def __init__(self, contamination=0.05):
        """
        Initialize the ML input validator.
       
        Args:
            contamination: The expected proportion of anomalies in the training data.
        """
        self.contamination = contamination
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        self.text_features = None
        self.trained = False
       
    def _extract_features(self, data):
        """
        Extract features from different data types.
       
        Args:
            data: DataFrame containing input data
           
        Returns:
            DataFrame with extracted features
        """
        features = data.copy()
       
        # For text fields, extract additional features
        for col in self.text_features:
            if col in features:
                # Add text length
                features[f'{col}_length'] = features[col].str.len()
               
                # Add special character count
                features[f'{col}_special_chars'] = features[col].apply(
                    lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', str(x)))
                )
               
                # Add uppercase ratio
                features[f'{col}_uppercase_ratio'] = features[col].apply(
                    lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
                )
               
                # Add digit ratio
                features[f'{col}_digit_ratio'] = features[col].apply(
                    lambda x: sum(1 for c in str(x) if c.isdigit()) / len(str(x)) if len(str(x)) > 0 else 0
                )
       
        return features
   
    def fit(self, training_data):
        """
        Train the ML model on known good data to learn normal patterns.
       
        Args:
            training_data: DataFrame containing valid input data
        """
        # Identify column types
        self.feature_names = training_data.columns.tolist()
        self.numeric_features = training_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = training_data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.text_features = [col for col in self.categorical_features
                             if training_data[col].map(lambda x: isinstance(x, str) and len(x) > 10).any()]
       
        # Extract features
        features_df = self._extract_features(training_data)
       
        # Prepare the column transformer for preprocessing
        transformers = []
       
        # Handle numeric features
        if self.numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, self.numeric_features))
       
        # Handle categorical features (excluding text features)
        categorical_short = [col for col in self.categorical_features if col not in self.text_features]
        if categorical_short:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_short))
       
        # Handle text-derived features
        text_derived = []
        for col in self.text_features:
            text_derived.extend([f'{col}_length', f'{col}_special_chars',
                                f'{col}_uppercase_ratio', f'{col}_digit_ratio'])
       
        if text_derived:
            text_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('text', text_transformer, text_derived))
       
        # Create the preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
       
        # Initialize and train the anomaly detection model
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', IsolationForest(contamination=self.contamination, random_state=42))
        ])
       
        self.model.fit(features_df)
        self.trained = True
       
        return self
   
    def validate(self, input_data, threshold=-0.5):
        """
        Validate input data using the trained model.
       
        Args:
            input_data: Single input or DataFrame of inputs to validate
            threshold: Anomaly score threshold, lower is more strict
           
        Returns:
            Dictionary with validation results
        """
        if not self.trained:
            return {"error": "Model not trained. Call fit() first."}
       
        # Convert single input to DataFrame if needed
        if not isinstance(input_data, pd.DataFrame):
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            else:
                return {"error": "Input must be a DataFrame or dictionary"}
       
        # Check if input has the expected columns
        missing_columns = set(self.feature_names) - set(input_data.columns)
        if missing_columns:
            return {"error": f"Missing columns: {missing_columns}"}
       
        # Extract features
        try:
            features_df = self._extract_features(input_data)
           
            # Get anomaly scores
            anomaly_scores = self.model.named_steps['classifier'].decision_function(
                self.model.named_steps['preprocessor'].transform(features_df)
            )
           
            # Determine which inputs are anomalous
            anomalies = (anomaly_scores < threshold)
           
            results = []
            for i, is_anomaly in enumerate(anomalies):
                result = {
                    "valid": not is_anomaly,
                    "anomaly_score": float(anomaly_scores[i]),
                    "index": i
                }
               
                # Add record-specific information
                if is_anomaly:
                    result["warning"] = "Input contains unusual patterns"
                   
                    # Identify which features might be causing the anomaly
                    if len(input_data) == 1:
                        unusual_features = self._identify_unusual_features(features_df.iloc[i])
                        if unusual_features:
                            result["unusual_features"] = unusual_features
               
                results.append(result)
           
            return {"results": results}
           
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
   
    def _identify_unusual_features(self, input_record):
        """
        Try to identify which features are unusual in an anomalous input.
       
        Args:
            input_record: Series containing a single input record
           
        Returns:
            List of potentially unusual features
        """
        unusual_features = []
       
        # Simple heuristic: identify features with extreme values
        for feature in self.numeric_features:
            if feature in input_record and abs(input_record[feature]) > 3:  # More than 3 std deviations
                unusual_features.append(feature)
       
        # Check text-derived features
        for col in self.text_features:
            feature_length = f'{col}_length'
            feature_special = f'{col}_special_chars'
            feature_upper = f'{col}_uppercase_ratio'
            feature_digit = f'{col}_digit_ratio'
           
            if feature_length in input_record and input_record[feature_length] > 1000:
                unusual_features.append(f"{col} (unusually long)")
           
            if feature_special in input_record and input_record[feature_special] > 10:
                unusual_features.append(f"{col} (many special characters)")
           
            if feature_upper in input_record and input_record[feature_upper] > 0.7:
                unusual_features.append(f"{col} (excessive uppercase)")
           
            if feature_digit in input_record and input_record[feature_digit] > 0.5:
                unusual_features.append(f"{col} (high digit ratio)")
       
        return unusual_features
   
    def save_model(self, filepath):
        """
        Save the trained model to a file.
       
        Args:
            filepath: Path to save the model
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
       
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "text_features": self.text_features,
            "contamination": self.contamination,
            "trained": self.trained
        }
       
        joblib.dump(model_data, filepath)
       
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.
       
        Args:
            filepath: Path to the saved model
           
        Returns:
            Loaded MLInputValidator instance
        """
        model_data = joblib.load(filepath)
       
        validator = cls(contamination=model_data["contamination"])
        validator.model = model_data["model"]
        validator.feature_names = model_data["feature_names"]
        validator.numeric_features = model_data["numeric_features"]
        validator.categorical_features = model_data["categorical_features"]
        validator.text_features = model_data["text_features"]
        validator.trained = model_data["trained"]
       
        return validator


# Example usage
if __name__ == "__main__":
    # Create some sample training data
    training_data = pd.DataFrame({
        "name": ["John Smith", "Jane Doe", "Robert Johnson", "Maria Garcia", "Wei Zhang"],
        "email": ["john@example.com", "jane@example.com", "robert@example.com",
                  "maria@example.com", "wei@example.com"],
        "age": [35, 28, 42, 31, 26],
        "income": [75000, 82000, 65000, 90000, 78000],
        "address": ["123 Main St, City, State", "456 Oak Ave, Town, State",
                    "789 Pine Rd, Village, State", "321 Elm Blvd, City, State",
                    "654 Maple Ln, Town, State"],
        "category": ["A", "B", "A", "C", "B"]
    })
   
    # Initialize and train the validator
    validator = MLInputValidator(contamination=0.1)
    validator.fit(training_data)
   
    # Save the trained model
    validator.save_model("input_validator_model.joblib")
   
    # Test with normal input
    normal_input = {
        "name": "David Brown",
        "email": "david@example.com",
        "age": 39,
        "income": 80000,
        "address": "987 Cedar St, City, State",
        "category": "A"
    }
   
    normal_result = validator.validate(normal_input)
    print("Normal input validation result:")
    print(json.dumps(normal_result, indent=2))
   
    # Test with anomalous input
    anomalous_input = {
        "name": "SUSPICIOUS_USER!!!!!",
        "email": "verylongemailaddresswithhiddenmaliciouscontentandmanyrandomcharacters123456789@example.com",
        "age": 999,
        "income": -50000,
        "address": "<script>alert('XSS')</script>",
        "category": "UNKNOWN"
    }
   
    anomalous_result = validator.validate(anomalous_input)
    print("\nAnomalous input validation result:")
    print(json.dumps(anomalous_result, indent=2))
