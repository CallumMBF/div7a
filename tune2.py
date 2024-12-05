import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import itertools
import logging
from pathlib import Path
import json
from datetime import datetime
import os
from typing import Dict, List
from app11 import LoanAssessmentApp

class Div7ATuner:
    def __init__(self, data_path: str, output_dir: str = "tuning_results"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.loan_app = LoanAssessmentApp(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        self.param_grid = {
            'debit_adjustment': [0.1, 0.2, 0.3, 0.4, 0.5],
            'probability_threshold': [0.5, 0.6, 0.7, 0.8],
            'batch_size': [10]
        }
        
        self._setup_logging()
        
    def _setup_logging(self):
        log_path = self.output_dir / f"tuning_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )

    def load_data(self) -> pd.DataFrame:
        logging.info(f"Loading data from {self.data_path}")
        df = pd.read_excel(self.data_path)
        
        required_cols = ['AccountName', 'Balance', 'SourceAccountType', 'AccountTypesName']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {', '.join(missing_cols)}")
        
        filtered_df = df[df.apply(
            lambda row: self.loan_app.requires_div7a_check(
                row['SourceAccountType'],
                row.get('AccountTypesName', '')
            ),
            axis=1
        )].copy()
        
        filtered_df['GroundTruth'] = 1  # All filtered accounts are warnings for tuning
        return filtered_df

    def process_batch(self, accounts: pd.DataFrame, params: Dict) -> List[float]:
        detection_results = self.loan_app.detect_loans_batch(
            accounts['AccountName'].tolist(),
            batch_size=params['batch_size']
        )
        
        results_map = {r['account_name']: r for r in detection_results}
        adjusted_probs = []
        
        for _, account in accounts.iterrows():
            result = results_map.get(account['AccountName'], {'probability': 0.0})
            prob = result['probability']
            
            if account['SourceAccountType'].lower() == 'liabilities' and account['Balance'] > 0:
                prob = min(1.0, prob + params['debit_adjustment'])
            
            adjusted_probs.append(prob)
        
        return adjusted_probs

    def evaluate_params(self, df: pd.DataFrame, params: Dict) -> Dict:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics = {'precision': [], 'recall': [], 'f1': []}
        
        for fold, (_, val_idx) in enumerate(kf.split(df), 1):
            val_df = df.iloc[val_idx]
            
            probabilities = self.process_batch(val_df, params)
            predictions = [1 if p >= params['probability_threshold'] else 0 for p in probabilities]
            actuals = [1] * len(val_df)
            
            metrics['precision'].append(precision_score(actuals, predictions))
            metrics['recall'].append(recall_score(actuals, predictions))
            metrics['f1'].append(f1_score(actuals, predictions))
            
        return {
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'f1': np.mean(metrics['f1']),
            'std_dev': np.std(metrics['f1'])
        }

    def run_tuning(self) -> pd.DataFrame:
        df = self.load_data()
        param_names = list(self.param_grid.keys())
        param_values = list(itertools.product(*self.param_grid.values()))
        
        results = []
        for i, values in enumerate(param_values, 1):
            params = dict(zip(param_names, values))
            logging.info(f"Testing combination {i}/{len(param_values)}: {params}")
            
            metrics = self.evaluate_params(df, params)
            results.append({**params, **metrics})
        
        results_df = pd.DataFrame(results)
        self.save_results(results_df)
        return results_df

    def save_results(self, results_df: pd.DataFrame):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_df.to_csv(self.output_dir / f"results_{timestamp}.csv", index=False)
        top_results = results_df.nlargest(10, 'f1')
        top_results.to_csv(self.output_dir / f"top_results_{timestamp}.csv", index=False)
        
        best_params = results_df.loc[results_df['f1'].idxmax()].to_dict()
        summary = {
            'timestamp': timestamp,
            'total_combinations': len(results_df),
            'best_parameters': best_params,
            'parameter_ranges': self.param_grid
        }
        
        with open(self.output_dir / f"summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Best parameters: {best_params}")

def main():
    data_path = "C:\\Users\\CallumMatchett\\python\\streamlit\\div7a\\MOCKTB2.xlsx"
    output_dir = "tuning_results2"
    
    tuner = Div7ATuner(data_path, output_dir)
    results = tuner.run_tuning()
    
    print("\nTop 10 Parameter Combinations:")
    print(results.nlargest(10, 'f1')[['debit_adjustment', 'probability_threshold', 
                                    'batch_size', 'f1', 'precision', 'recall']])
    """
    Precision: Of all accounts flagged as Division 7A, what percentage actually are? (False positives are costly as they waste auditor time)
    Recall: Of all actual Division 7A accounts, what percentage did we catch? (False negatives are costly as they miss compliance issues)
    F1 Score: Harmonic mean of precision and recall (F1 = 2 * (precision * recall)/(precision + recall)). Balances the tradeoff between catching all Division 7A loans vs. minimizing false alarms.
    We optimize for F1 score to find parameters that achieve both high precision and recall.
    """
if __name__ == "__main__":
    main()