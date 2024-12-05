# Import required libraries
import streamlit as st
import pandas as pd
from openai import OpenAI
import os

class LoanAssessmentApp:
    def __init__(self, openai_api_key=None):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Hardcoded rules for account combinations that require Div 7A checking 
        self.check_combinations = {
            ('Assets', 'Loans to Members'),
            ('Assets', 'Other Assets'),
            ('Assets', 'Other Liabilities'),
            ('Equity', 'Adjustments to Retained Earnings'),
            ('Liabilities', 'Loans from Members'),
            ('Liabilities', 'Other Assets'),
            ('Liabilities', 'Other Liabilities'),
            ('Liabilities', 'Borrowings'),
            ('Other', 'Retained Earnings')
        }

        self.loan_detection_prompt = """
            Analyze the following accounts to determine if they represent Division 7A loans or payments.
            For each account, provide a probability score from 0.0 to 1.0 and a brief reasoning.

            Key Division 7A Indicators:
            1. Direct indicators:
                - Shareholders or their associates
                - Directors including Management or their associates
                - Associate entities or individuals
                - Related parties
            2. Masked/indirect indicators:
                - Accounts that could potentially hide related party transactions
                - Generic business terms that might mask personal use
                - Vague or ambiguous account names that could conceal Division 7A arrangements
                - Terms suggesting temporary or special arrangements
                - Personal or individual transactions mixed with business purposes

            Assignment Rules:
            - 1.0 = Definitely Division 7A related
            - 0.8 = Very likely Division 7A related (clear ownership/relationship terms)
            - 0.6 = Suspicious - vaguely worded but likely masking Division 7A
            - 0.4 = Some Division 7A indicators but unclear
            - 0.2 = Unlikely Division 7A related but requires attention
            - 0.0 = Definitely not Division 7A related

            Accounts to analyze:
            {account_list}

            Provide responses in the following format for each account:
            Account: [account name]
            Probability: [0.0-1.0]
            Reasoning: [One clear sentence explaining the probability assignment and any masking concerns]
            ---
            """
    
    def requires_div7a_check(self, source_account_type, account_types_name):
        """Check if the account combination requires Division 7A analysis"""
        return (source_account_type, account_types_name) in self.check_combinations
    
    def detect_loans_batch(self, accounts, batch_size=5):
        """Process multiple accounts in batches for loan detection"""
        results = []
        
        # Process accounts in batches
        for i in range(0, len(accounts), batch_size):
            batch = accounts[i:i + batch_size]
            
            # Format account list for the prompt
            account_list = "\n".join([f"- {account}" for account in batch])
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a financial classification expert."},
                        {"role": "user", "content": self.loan_detection_prompt.format(account_list=account_list)}
                    ],
                    max_tokens=100,
                    temperature=0
                )
                
                # Parse the batch response
                result_text = response.choices[0].message.content.strip()
                account_results = self._parse_batch_response(result_text)
                
                # Match results with original accounts
                for account, result in zip(batch, account_results):
                    results.append({
                        'account_name': account,
                        'probability': result.get('probability', 0.0),
                        'reasoning': result.get('reasoning', 'No specific reasoning provided')
                    })
                    
            except Exception as e:
                # Handle errors by adding default results for the batch
                for account in batch:
                    results.append({
                        'account_name': account,
                        'probability': 0.0,
                        'reasoning': f'Batch Processing Error: {str(e)}'
                    })
        
        return results
    
    def _parse_batch_response(self, response_text):
        """Parse the batch response from LLM into individual results"""
        results = []
        current_result = {}
        
        for line in response_text.split('\n'):
            line = line.strip()
            if not line or line == '---':
                if current_result:
                    results.append(current_result)
                    current_result = {}
                continue
                
            if line.startswith('Account:'):
                if current_result:
                    results.append(current_result)
                current_result = {}
                current_result['account_name'] = line.split(':', 1)[1].strip()
            elif line.startswith('Probability:'):
                try:
                    current_result['probability'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    current_result['probability'] = 0.0
            elif line.startswith('Reasoning:'):
                current_result['reasoning'] = line.split(':', 1)[1].strip()
        
        if current_result:
            results.append(current_result)
            
        return results

def analyze_account(loan_app, row):
    """Modified account analysis to use batch processing results"""
    result = {
        'Account': row['AccountName'],
        'Balance': row['Balance'],
        'SourceAccountType': row['SourceAccountType'],
        'AccountTypesName': row.get('AccountTypesName', ''),
        'Loan Detection': {
            'Probability': 0.0,
            'Reasoning': None
        },
        'Final_Probability': 0.0,
        'Summary': None,
        'Div7A Status': None,
        'Requires_Warning': False
    }
    
    # Check if we have loan detection results for this account
    if hasattr(row, 'loan_detection_result'):
        if isinstance(row.loan_detection_result, dict):
            result['Loan Detection']['Probability'] = row.loan_detection_result.get('probability', 0.0)
            result['Loan Detection']['Reasoning'] = row.loan_detection_result.get('reasoning', 'No specific reasoning provided')
    
    # Get Division 7A status
    probability, status, needs_warning = check_div7a_status(
        row['SourceAccountType'],
        row['Balance'],
        row.get('WorkpaperRecordsName', ''),
        result['Loan Detection']['Probability']
    )
    
    result['Final_Probability'] = probability
    result['Div7A Status'] = status
    result['Requires_Warning'] = needs_warning
    result['Summary'] = f"Div 7A Probability: {probability:.1%} - {status}"
    
    return result

def check_div7a_status(account_type, balance, workpaper_record, initial_probability):
    """
    Determine Division 7A probability and status based on account characteristics
    """
    # Handle null/missing values
    account_type = str(account_type).lower() if account_type and not pd.isna(account_type) else ''
    workpaper_record = str(workpaper_record).lower() if workpaper_record and not pd.isna(workpaper_record) else ''
    
    # Initialize probability from LLM
    probability = initial_probability
    
    # Check for existing Division 7A documentation
    if any(term in workpaper_record for term in ['div 7a', 'division 7a', 'div. 7a', 'div7a']):
        return 1.0, "Div 7A reconciliation worksheet already attached", False
    
    # Adjust probability based on account type and balance
    if account_type == 'liabilities':
        if balance > 0:  # Debited liability
            probability = min(probability + 0.2, 1)  # Increase probability significantly for debit balance
            
            if probability >= 0.6:
                status = "Liability account with debit balance - high probability of Division 7A implication"
            else:
                status = "Liability account with debit balance - low probability of Division 7A implication"
        else:  # Credited liability
            probability = min(probability, 0)  # no div 7A
            status = "Liability account with credit balance - not likely to be a Division 7A loan"    
    elif account_type in ('assets', 'equity'):
        if probability >= 0.6:
            status = "Potential Division 7A loan - review recommended"
        else:
            status = "Not likely to be a Division 7A loan"
    else:
        status = "Account type unclear - review needed"
    
    # Determine if warning is needed based on probability
    needs_warning = probability >= 0.6
    
    return probability, status, needs_warning

def main():
    """Modified main function to use batch processing"""
    st.title("Div 7A Loan Assessment Tool")
    
    st.sidebar.header("Configuration")
    openai_api_key = os.getenv('OPENAI_API_KEY')

    trial_balance_file = st.sidebar.file_uploader("Upload Trial Balance", type=['xlsx'])
    
    if st.sidebar.button("Analyse Accounts") and trial_balance_file and openai_api_key:
        loan_app = LoanAssessmentApp(openai_api_key)
        
        # Show loading message
        st.info("Processing trial balance file...")
        
        trial_balance_df = pd.read_excel(trial_balance_file)
        
        required_columns = ['AccountName', 'Balance', 'SourceAccountType', 'AccountTypesName']
        missing_columns = [col for col in required_columns if col not in trial_balance_df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in trial balance: {', '.join(missing_columns)}")
            return
        
        progress_bar = st.progress(0)
        
        # Get accounts that need Div 7A checking
        accounts_to_check = trial_balance_df[
            trial_balance_df.apply(lambda row: loan_app.requires_div7a_check(
                row['SourceAccountType'], 
                row.get('AccountTypesName', '')
            ), axis=1)
        ]
        
        st.write(f"Found {len(accounts_to_check)} accounts to check for Division 7A")
        
        # Batch process the loan detection
        results = []
        if not accounts_to_check.empty:
            loan_detection_results = loan_app.detect_loans_batch(
                accounts_to_check['AccountName'].tolist()
            )
            
            # Create a mapping of results
            results_map = {r['account_name']: r for r in loan_detection_results}
            
            # Add results back to the DataFrame
            trial_balance_df['loan_detection_result'] = trial_balance_df['AccountName'].map(
                lambda x: results_map.get(x, {'probability': 0.0, 'reasoning': 'No analysis required'})
            )
        
            for index, row in trial_balance_df.iterrows():
                progress_bar.progress(int((index + 1) / len(trial_balance_df) * 100))
                account_result = analyze_account(loan_app, row)
                results.append(account_result)

            # Display results
            st.header("Loan Assessment Results")
            
            # Sort results by warning status and probability
            sorted_results = sorted(results, 
                                 key=lambda x: (x['Requires_Warning'], x['Final_Probability']), 
                                 reverse=True)

            for result in sorted_results:
                if result['Requires_Warning']:
                    status_color = 'red'
                    status_emoji = "⚠️"
                elif result['Final_Probability'] >= 0.6:
                    status_color = 'orange'
                    status_emoji = "ℹ️"
                else:
                    status_color = 'green'
                    status_emoji = "✅"

                with st.expander(f"{status_emoji} {result['Account']} - Probability: {result['Final_Probability']:.1%}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Account Details:**")
                        st.write(f"Account Type: {result['SourceAccountType']}")
                        st.write(f"Balance: ${result['Balance']:,.2f}")
                        
                    with col2:
                        st.write("**Analysis Results:**")
                        st.write(f"Status: {result['Div7A Status']}")
                        st.write(f"Warning Required: {result['Requires_Warning']}")
                    
                    st.write("**LLM Analysis:**")
                    if result['Loan Detection']['Reasoning']:
                        st.write(result['Loan Detection']['Reasoning'])

            # Add download button for results
            results_df = pd.DataFrame(results)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Full Results CSV",
                data=csv,
                file_name="div7a_analysis_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No accounts found that require Division 7A checking.")
    else:
        st.write("Please upload a trial balance file and click 'Analyse Accounts' to begin.")

if __name__ == "__main__":
    main()