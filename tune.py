import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime
from pathlib import Path
from app11 import LoanAssessmentApp, analyze_account  # Make sure to import both
import os 

class HyperparameterTuner:
    def __init__(self, loan_app, data_file: str, output_dir: str = "tuning_results"):
        self.loan_app = loan_app
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Default hyperparameter ranges
        self.hyperparameters = {
            "vague_adjustment": [0.2],
            "debit_liability_adjustment": [0.2],
            "probability_threshold": [0.5]  # Added threshold as a tunable parameter
        }
        
    def _setup_logging(self):
        """Configure logging for the tuning process"""
        log_file = self.output_dir / f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate the input data"""
        try:
            df = pd.read_excel(self.data_file)
            required_columns = ['AccountName', 'Balance', 'SourceAccountType', 
                            'AccountTypesName']  # Removed GroundTruth from required columns
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Add GroundTruth column with "Warning" for all records
            df['GroundTruth'] = "Warning"
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
            
    def evaluate_parameters(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Evaluate a single set of parameters using k-fold cross-validation"""
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics = {
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            val_df = df.iloc[val_idx]
            fold_predictions = self._get_predictions(val_df, params)
            
            true_labels = [int(label == "Warning") for label in val_df['GroundTruth']]
            predicted_labels = [int(prob >= params['probability_threshold']) 
                              for prob in fold_predictions]
            
            metrics['precision'].append(precision_score(true_labels, predicted_labels))
            metrics['recall'].append(recall_score(true_labels, predicted_labels))
            metrics['f1'].append(f1_score(true_labels, predicted_labels))
            
        return {
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'f1': np.mean(metrics['f1']),
            'std_dev': np.std(metrics['f1'])
        }
        
    def _get_predictions(self, df: pd.DataFrame, params: Dict) -> List[float]:
        """Get predictions for a dataset using given parameters"""
        predictions = []
        
        for _, row in df.iterrows():
            result = analyze_account(self.loan_app, row)
            prob = result['Loan Detection']['Probability']
            
            # Apply parameter adjustments
            if 'somewhat vague' in result['Loan Detection']['Reasoning'].lower():
                prob = min(prob + params['vague_adjustment'], 1.0)
                
            if row['SourceAccountType'].lower() == 'liabilities' and row['Balance'] > 0:
                prob = min(prob + params['debit_liability_adjustment'], 1.0)
                
            predictions.append(prob)
            
        return predictions
        
    def run_tuning(self) -> pd.DataFrame:
        """Run the complete hyperparameter tuning process"""
        logging.info("Starting hyperparameter tuning")
        
        try:
            df = self.load_and_validate_data()
            
            # Generate parameter combinations
            param_names = list(self.hyperparameters.keys())
            param_values = list(itertools.product(*self.hyperparameters.values()))
            
            results = []
            total_combinations = len(param_values)
            
            for i, values in enumerate(param_values, 1):
                params = dict(zip(param_names, values))
                logging.info(f"Testing parameters {i}/{total_combinations}: {params}")
                
                metrics = self.evaluate_parameters(df, params)
                
                results.append({
                    **params,
                    **metrics
                })
                
            results_df = pd.DataFrame(results)
            self._save_results(results_df)
            
            return results_df
            
        except Exception as e:
            logging.error(f"Error during tuning: {str(e)}")
            raise
            
    def _save_results(self, results_df: pd.DataFrame):
        """Save tuning results and generate summary"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_path = self.output_dir / f"tuning_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save top results summary
        top_results = results_df.nlargest(10, 'f1')
        top_results_path = self.output_dir / f"top_results_{timestamp}.csv"
        top_results.to_csv(top_results_path, index=False)
        
        # Generate and save summary report
        summary = {
            'timestamp': timestamp,
            'total_combinations_tested': len(results_df),
            'best_parameters': results_df.iloc[results_df['f1'].argmax()].to_dict(),
            'parameter_ranges': self.hyperparameters
        }
        
        summary_path = self.output_dir / f"tuning_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logging.info(f"Results saved to {self.output_dir}")

def main():
    # Example usage
    from app11 import LoanAssessmentApp
    
    loan_app = LoanAssessmentApp(openai_api_key=os.getenv("OPENAI_API_KEY"))
    tuner = HyperparameterTuner(loan_app, "MOCKTB2.xlsx")
    
    try:
        results_df = tuner.run_tuning()
        print("\nTop 10 Parameter Combinations:")
        print(results_df.nlargest(10, 'f1')[['vague_adjustment', 'debit_liability_adjustment', 
                                           'probability_threshold', 'f1', 'precision', 'recall']])
    except Exception as e:
        print(f"Error during tuning: {str(e)}")

if __name__ == "__main__":
    main()