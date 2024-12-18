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
            'probability_threshold': [0.4, 0.5, 0.6],
            'max_tokens': [500, 750, 1000],
            'batch_size': [50, 75, 100]  # Smaller batch sizes
        }
        
        # Track misclassified accounts across folds
        self.misclassified_accounts = {}
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
        
        # Log initial counts
        logging.info(f"Initial data: {len(df)} total accounts, {df['GroundTruth'].sum()} Division 7A accounts")
        
        # Clean any leading/trailing spaces in relevant columns
        df['SourceAccountType'] = df['SourceAccountType'].str.strip()
        df['AccountTypesName'] = df['AccountTypesName'].str.strip()
        
        # Get filtered out accounts
        filtered_out = df[~df.apply(
            lambda row: self.loan_app.requires_div7a_check(
                row['SourceAccountType'],
                row.get('AccountTypesName', '')
            ),
            axis=1
        )]
        
        # Save filtered out accounts to CSV for inspection
        filtered_out.to_csv('filtered_out_accounts.csv', index=False)
        logging.info(f"Filtered out {len(filtered_out)} accounts. Details saved to 'filtered_out_accounts.csv'")
        
        # Get filtered in accounts
        filtered_df = df[df.apply(
            lambda row: self.loan_app.requires_div7a_check(
                row['SourceAccountType'],
                row.get('AccountTypesName', '')
            ),
            axis=1
        )].copy()
        
        # Add logging to see distribution of true labels
        pos_count = filtered_df['GroundTruth'].sum()
        total_count = len(filtered_df)
        logging.info(f"After filtering: {pos_count} Division 7A accounts out of {total_count} total accounts")
        
        # Log some examples of filtered out Div7A accounts
        filtered_out_div7a = filtered_out[filtered_out['GroundTruth'] == 1]
        if not filtered_out_div7a.empty:
            logging.info("\nExamples of Division 7A accounts that were filtered out:")
            for _, row in filtered_out_div7a.head().iterrows():
                logging.info(f"Account: {row['AccountName']}")
                logging.info(f"SourceAccountType: '{row['SourceAccountType']}'")
                logging.info(f"AccountTypesName: '{row['AccountTypesName']}'\n")

        return filtered_df

    
    def process_batch(self, accounts: pd.DataFrame, params: Dict) -> List[float]:
        # Override the max_tokens in the loan_app's detect_loans_batch method
        original_detect_loans_batch = self.loan_app.detect_loans_batch

        def modified_detect_loans_batch(account_list, batch_size):
            detection_results = []
            
            # Process accounts in smaller batches
            for i in range(0, len(account_list), batch_size):
                batch = account_list[i:i + batch_size]
                
                # Format account list for the prompt
                account_list_str = "\n".join([f"- {account}" for account in batch])
                
                try:
                    response = self.loan_app.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a financial classification expert. For EACH account in the list, provide an assessment."},
                            {"role": "user", "content": f"{self.loan_app.loan_detection_prompt.format(account_list=account_list_str)}\nIMPORTANT: Provide an assessment for EVERY account in the list."}
                        ],
                        max_tokens=params['max_tokens'],
                        temperature=0
                    )
                    
                    result_text = response.choices[0].message.content.strip()
                    batch_results = self.loan_app._parse_batch_response(result_text)
                    
                    # Verify we got results for all accounts in the batch
                    received_accounts = {r['account_name'] for r in batch_results}
                    missing_accounts = set(batch) - received_accounts
                    
                    # Add default results for any missing accounts
                    for account in missing_accounts:
                        logging.warning(f"API didn't return result for {account}, using default")
                        batch_results.append({
                            'account_name': account,
                            'probability': 0.0,
                            'reasoning': 'No assessment provided by API'
                        })
                        
                    detection_results.extend(batch_results)
                    
                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    # Handle errors by adding default results for the batch
                    for account in batch:
                        detection_results.append({
                            'account_name': account,
                            'probability': 0.0,
                            'reasoning': f'Error: {str(e)}'
                        })
            
            return detection_results
        
        # Temporarily replace the detect_loans_batch method
        self.loan_app.detect_loans_batch = modified_detect_loans_batch
        
        # Process the accounts with the modified method
        detection_results = self.loan_app.detect_loans_batch(
            accounts['AccountName'].tolist(),
            batch_size=params['batch_size']
        )
        
        # Restore the original method
        self.loan_app.detect_loans_batch = original_detect_loans_batch
        
        # Add debug logging for detection_results
        logging.info(f"Detection results structure: {detection_results[:2]}")  # Show first 2 results
        
        results_map = {r['account_name']: r for r in detection_results}
        adjusted_probs = []
        
        for _, account in accounts.iterrows():
            account_name = account['AccountName']
            result = results_map.get(account_name)
            
            if result is None:
                logging.error(f"No result found for account: {account_name}")
                prob = 0.0
            else:
                try:
                    prob = result.get('probability', 0.0)
                    logging.debug(f"Account: {account_name}, Result structure: {result}")
                except Exception as e:
                    logging.error(f"Error processing result for {account_name}: {str(e)}")
                    logging.error(f"Result structure: {result}")
                    prob = 0.0
            
            if account['SourceAccountType'].lower() == 'liabilities' and account['Balance'] > 0:
                prob = min(1.0, prob + params['debit_adjustment'])
            
            adjusted_probs.append(prob)
            
            # Add debugging logs
            logging.debug(f"Account: {account_name}")
            logging.debug(f"Initial probability: {prob}")
            logging.debug(f"Adjusted probability: {prob}")
        
        # Add summary statistics
        logging.info(f"Batch statistics:")
        logging.info(f"Average probability: {np.mean(adjusted_probs):.3f}")
        logging.info(f"Max probability: {max(adjusted_probs):.3f}")
        logging.info(f"Min probability: {min(adjusted_probs):.3f}")

        detection_results = self.loan_app.detect_loans_batch(
        accounts['AccountName'].tolist(),
        batch_size=params['batch_size']
        )
    
        # Add this debug logging right after getting detection_results
        logging.info("=== Detection Results ===")
        logging.info(f"Number of accounts sent: {len(accounts)}")
        logging.info(f"Number of results received: {len(detection_results)}")
        logging.info("First few detection results:")
        for result in detection_results[:5]:
            logging.info(f"Result: {result}")
        
        results_map = {r['account_name']: r for r in detection_results}
        logging.info("=== Results Map ===")
        logging.info(f"Number of mapped results: {len(results_map)}")
        logging.info("First few mapped accounts:")
        for key in list(results_map.keys())[:5]:
            logging.info(f"Mapped account: {key}")
            
            return adjusted_probs


    def track_misclassifications(self, accounts: pd.DataFrame, predictions: List[int], 
                               actuals: List[int], params: Dict):
        """Track consistently misclassified accounts for each parameter combination"""
        param_key = (f"debit_{params['debit_adjustment']}_thresh_{params['probability_threshold']}"
                    f"_tokens_{params['max_tokens']}")
        
        if param_key not in self.misclassified_accounts:
            self.misclassified_accounts[param_key] = {}
            
        for idx, (pred, actual) in enumerate(zip(predictions, actuals)):
            account_name = accounts.iloc[idx]['AccountName']
            if actual == 1 and pred == 0:  # False negative
                if account_name not in self.misclassified_accounts[param_key]:
                    self.misclassified_accounts[param_key][account_name] = {
                        'misclassification_count': 0,
                        'total_appearances': 0,
                        'account_details': accounts.iloc[idx].to_dict()
                    }
                self.misclassified_accounts[param_key][account_name]['misclassification_count'] += 1
            
            if actual == 1:  # Track total appearances for true positives
                if account_name not in self.misclassified_accounts[param_key]:
                    self.misclassified_accounts[param_key][account_name] = {
                        'misclassification_count': 0,
                        'total_appearances': 0,
                        'account_details': accounts.iloc[idx].to_dict()
                    }
                self.misclassified_accounts[param_key][account_name]['total_appearances'] += 1

    def evaluate_params(self, df: pd.DataFrame, params: Dict) -> Dict:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        metrics = {'precision': [], 'recall': [], 'f1': []}
        
        for fold, (_, val_idx) in enumerate(kf.split(df), 1):
            val_df = df.iloc[val_idx]
            
            probabilities = self.process_batch(val_df, params)
            predictions = [1 if p >= params['probability_threshold'] else 0 for p in probabilities]
            actuals = val_df['GroundTruth'].tolist()
            
            # Track misclassifications for this fold
            self.track_misclassifications(val_df, predictions, actuals, params)
            
            metrics['precision'].append(precision_score(actuals, predictions, zero_division=0))
            metrics['recall'].append(recall_score(actuals, predictions, zero_division=0))
            metrics['f1'].append(f1_score(actuals, predictions, zero_division=0))
        
        return {
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'f1': np.mean(metrics['f1']),
            'std_dev': np.std(metrics['f1'])
        }
    
    def analyze_misclassifications(self):
        """Analyze and save consistently misclassified accounts"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_path = self.output_dir / f"misclassification_analysis_{timestamp}"
        analysis_path.mkdir(exist_ok=True)
        
        for param_key, accounts in self.misclassified_accounts.items():
            records = []
            for account_name, data in accounts.items():
                misclassification_rate = (data['misclassification_count'] / 
                                        data['total_appearances'] if data['total_appearances'] > 0 else 0)
                
                record = {
                    'AccountName': account_name,
                    'MisclassificationRate': misclassification_rate,
                    'MisclassificationCount': data['misclassification_count'],
                    'TotalAppearances': data['total_appearances'],
                    **data['account_details']
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            # Filter for consistently misclassified accounts
            consistent_misclassifications = df[df['MisclassificationRate'] >= 0.8].sort_values(
                'MisclassificationRate', ascending=False)
            
            # Save results
            df.to_csv(analysis_path / f"{param_key}_all_misclassifications.csv", index=False)
            consistent_misclassifications.to_csv(
                analysis_path / f"{param_key}_consistent_misclassifications.csv", index=False)
            
            # Log summary statistics
            logging.info(f"\nMisclassification Analysis for {param_key}:")
            logging.info(f"Total accounts analyzed: {len(df)}")
            logging.info(f"Consistently misclassified accounts: {len(consistent_misclassifications)}")
            logging.info("Top 5 most consistently misclassified accounts:")
            logging.info(consistent_misclassifications.head()[
                ['AccountName', 'MisclassificationRate', 'Balance', 'SourceAccountType']
            ].to_string())


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
        
        # Analyze misclassifications after all evaluations
        self.analyze_misclassifications()
        
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