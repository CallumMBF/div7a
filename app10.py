# Import required libraries
import streamlit as st  # Used for creating web applications
import pandas as pd    # Used for data manipulation and analysis
import openai         # OpenAI API client
from openai import OpenAI  # Specific OpenAI client class
import os            # Operating system interface for environment variables

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
            Analyze if the account '{account_name}' represents a Division 7A loan or payment.

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
            Assign a probability score from 0.0 to 1.0 where:
            - 1.0 = Definitely Division 7A related
            - 0.8 = Very likely Division 7A related (clear ownership/relationship terms)
            - 0.6 = Suspicious - vaguely worded but likely masking Division 7A
            - 0.4 = Some Division 7A indicators but unclear
            - 0.2 = Unlikely Division 7A related but requires attention
            - 0.0 = Definitely not Division 7A related

            Consider:
            - Is the account name intentionally vague or generic?
            - Could this be masking a related party arrangement?
            - Does the terminology suggest mixed personal/business use?
            - Are there indirect hints of shareholder/director benefits?

            Response Format:
            Probability: [0.0-1.0]
            Reasoning: [One clear sentence explaining the probability assignment and any masking concerns]
            """
    
    def requires_div7a_check(self, source_account_type, account_types_name):
        """
        Check if the account combination requires Division 7A analysis
        Returns True if Reserved = Y for this combination
        """
        return (source_account_type, account_types_name) in self.check_combinations
    
    def detect_loan(self, account_name):
        """Modified to handle probability-based detection"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial classification expert."},
                    {"role": "user", "content": self.loan_detection_prompt.format(account_name=account_name)}
                ],
                max_tokens=200,
                temperature=0
            )
            
            result_text = response.choices[0].message.content.strip()
            lines = result_text.split('\n')
            
            probability = float(lines[0].split(':')[1].strip()) if ':' in lines[0] else 0.0
            reasoning = lines[1].split(':')[1].strip() if len(lines) > 1 and ':' in lines[1] else 'No specific reasoning provided'
            
            return {
                'account_name': account_name,
                'probability': probability,
                'reasoning': reasoning
            }
        except Exception as e:
            return {
                'account_name': account_name,
                'probability': 0.0,
                'reasoning': f'Loan Detection Error: {str(e)}'
            }


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
    if any(term in workpaper_record for term in ['div 7a', 'division 7a']):
        return 1.0, "Div 7A reconciliation worksheet already attached", False
    
    # Adjust probability based on account type and balance
    if account_type == 'liabilities':
        if balance > 0:  # Debited liability
            probability = min(probability + 0.4, 1)  # Increase probability significantly for debit balance
            
            if probability >= 0.5:
                status = "Liability account with debit balance - high probability of Division 7A implication"
            else:
                status = "Liability account with debit balance - low probability of Division 7A implication"
        else:  # Credited liability
            probability = min(probability, 0)  # no div 7A
            status = "Liability account with credit balance - not likely to be a Division 7A loan"    
    elif account_type in ('assets', 'equity'):
        if probability >= 0.5:
            status = "Potential Division 7A loan - review recommended"
        else:
            status = "Not likely to be a Division 7A loan"
    else:
        status = "Account type unclear - review needed"
    
    # Determine if warning is needed based on probability
    needs_warning = probability >= 0.5
    
    return probability, status, needs_warning

def analyze_account(loan_app, row):
    """Modified account analysis to include pre-check and vague term probability adjustment"""
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
    
    # Check if account combination requires Division 7A analysis
    if loan_app.requires_div7a_check(row['SourceAccountType'], row.get('AccountTypesName', '')):
        # Proceed with LLM-based detection for combinations marked as Reserved = Y
        loan_detection = loan_app.detect_loan(row['AccountName'])
        initial_probability = float(loan_detection.get('probability', 0.0))
        reasoning = loan_detection['reasoning']
        
        # New check: Increase probability by 0.1 if reasoning contains 'somewhat vague'
        if 'somewhat vague' in reasoning.lower():
            adjusted_probability = min(initial_probability + 0.1, 1.0)  # Ensure we don't exceed 1.0
            result['Loan Detection']['Probability'] = adjusted_probability
        else:
            result['Loan Detection']['Probability'] = initial_probability
            
        result['Loan Detection']['Reasoning'] = reasoning
        
        # Check Division 7A status with adjusted probability
        probability, status, needs_warning = check_div7a_status(
            row['SourceAccountType'],
            row['Balance'],
            row.get('WorkpaperRecordsName', ''),
            result['Loan Detection']['Probability']
        )
    else:
        # Skip LLM call for non-reserved combinations
        probability = 0.0
        status = "Not likely to be a Division 7A loan - Standard account type"
        needs_warning = False
        result['Loan Detection']['Reasoning'] = "Standard account type combination - no Division 7A analysis required"
    
    result['Final_Probability'] = probability
    result['Div7A Status'] = status
    result['Requires_Warning'] = needs_warning
    result['Summary'] = f"Div 7A Probability: {probability:.1%} - {status}"
    
    return result

def main():
    """Main function to run the Streamlit web application"""
    st.title("Div 7A Loan Assessment Tool")
    
    st.sidebar.header("Configuration")
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # Add file upload widget for trial balance
    trial_balance_file = st.sidebar.file_uploader("Upload Trial Balance", type=['xlsx'])
    
    results = []
    
    if st.sidebar.button("Analyse Accounts") and trial_balance_file and openai_api_key:
        # Initialize loan assessment application
        loan_app = LoanAssessmentApp(openai_api_key)
        
        # Read trial balance
        trial_balance_df = pd.read_excel(trial_balance_file)
        
        # Verify required columns
        required_columns = ['AccountName', 'Balance', 'SourceAccountType', 'AccountTypesName']
        missing_columns = [col for col in required_columns if col not in trial_balance_df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in trial balance: {', '.join(missing_columns)}")
            return
        
        progress_bar = st.progress(0)
        
        for index, row in trial_balance_df.iterrows():
            progress_bar.progress(int((index + 1) / len(trial_balance_df) * 100))
            account_result = analyze_account(loan_app, row)
            results.append(account_result)
    
    if results:
        results_df = pd.DataFrame(results)
        st.header("Loan Assessment Results")
        
        def get_sort_priority(result):
            if result['Requires_Warning']:
                return 0
            elif 0.5 < result['Loan Detection']['Probability'] < 0.8:
                return 1
            else:
                return 2
        
        sorted_results = sorted(results, key=get_sort_priority)

        for result in sorted_results:
            if result['Requires_Warning']:
                status_color = 'red'
                status_emoji = "⚠️"
            elif 0.5 < result['Loan Detection']['Probability'] < 0.8:
                status_color = 'orange'
                status_emoji = "ℹ️"
            else:
                status_color = 'green'
                status_emoji = "✅"

            with st.expander(f"{status_emoji} {result['Account']} - {result['Summary']}", expanded=False):
                st.markdown(f"**Summary:** <font color='{status_color}'>{result['Summary']}</font>", unsafe_allow_html=True)
                st.write(f"**Account Type:** {result['SourceAccountType']} - {result['AccountTypesName']}")
                
                st.subheader("Loan Detection")
                st.write(f"**Probability:** {result['Loan Detection']['Probability']:.1%}")
                st.write(f"**Reasoning:** {result['Loan Detection']['Reasoning']}")
                
                if result['Div7A Status']:
                    st.subheader("Div 7A Status")
                    st.write(result['Div7A Status'])

        sorted_results_df = pd.DataFrame(sorted_results)
        st.download_button(
            label="Download Full Results",
            data=sorted_results_df.to_csv(index=False),
            file_name="loan_assessment_results.csv",
            mime="text/csv"
        )
    else:
        st.write("")

if __name__ == "__main__":
    main()