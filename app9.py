# Import required libraries
import streamlit as st  # Used for creating web applications
import pandas as pd    # Used for data manipulation and analysis
import openai         # OpenAI API client
from openai import OpenAI  # Specific OpenAI client class
import os            # Operating system interface for environment variables

class LoanAssessmentApp:
    def __init__(self, openai_api_key=None):
        """Initialize the Loan Assessment Application
        
        This class handles the core functionality of assessing whether accounts
        might be subject to Division 7A tax provisions in Australia.
        """
        # Initialize OpenAI client with API key from environment variables
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Define the prompt template for the AI model
        # This prompt helps identify accounts that might be loans subject to Division 7A
        self.loan_detection_prompt = """
            Analyze if the account '{account_name}' represents a Division 7A loan or payment.

            Key Division 7A Indicators:
            1. Loans/advances/debts involving:
                - Shareholders or their associates
                - Directors including Management or their associates
                - Associate entities or individuals
                - Related parties
                - Business owners or key management
                - Personal or individual names
            2. Trust distributions that remain unpaid
            3. Loans between related private companies
            4. Any financial arrangement benefiting shareholders/associates
            5. High-risk terminology:
                - Development loans/funds
                - Temporary or interim funding
                - Special purpose accounts
                - Investment holdings
                - Personal transactions
                - Consulting fees (when payable to related parties)
                - Distribution accounts
                - Special drawing arrangements
                - Clearing accounts involving personal items

            Assignment Rules:
            Assign a probability score from 0.0 to 1.0 where:
            - 1.0 = Definitely Division 7A related
            - 0.8 = Very likely Division 7A related
            - 0.6 = Probably Division 7A related
            - 0.4 = Possibly but unlikely Division 7A related
            - 0.2 = Unlikely Division 7A related
            - 0.0 = Definitely not Division 7A related

            Scoring Guidelines:
            - Score 1.0 if contains: "Shareholder", "Director", "Associate", "Related Party"
            - Score 0.8 if contains: "Loan", "Advance", "Drawing" with ownership implications
            - Score 0.4 if contains:
                - "Temporary Funding", "Special Purpose"
                - "Investment Holdings"
                - "Consulting Fees"
                - "Distribution Account"
                - "Clearing Account"
                - "Special Drawing"
            - Score 0.2 if contains ambiguous financial terms
            - Score 0.0 if clearly unrelated (e.g., trade creditors, GST)

            Additional Context:
                - Consider account names with personal names or initials as high risk
                - Business development and corporate funding accounts often mask related party transactions
                - Temporary or special purpose accounts may indicate attempted Division 7A avoidance
                - Consulting and management fees can be disguised distributions
                - Distribution accounts may represent unpaid trust distributions
                - Clearing accounts with personal elements suggest mixed usage

            Response Format:
            Probability: [0.0-1.0]
            Reasoning: [One clear sentence explaining the probability assignment]
            """

    
    def detect_loan(self, account_name):
        """Modified to handle probability-based detection"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial classification expert."},
                    {"role": "user", "content": self.loan_detection_prompt.format(account_name=account_name)}
                ],
                max_tokens=200,
                temperature=0
            )
            
            result_text = response.choices[0].message.content.strip()
            lines = result_text.split('\n')
            
            # Parse probability and reasoning
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
    
    Args:
        account_type (str): Type of account (assets/liabilities)
        balance (float): Account balance
        workpaper_record (str): Any existing workpaper documentation
        initial_probability (float): Initial probability from LLM analysis
        
    Returns:
        tuple: (final_probability (float), status (str), needs_warning (bool))
    """
    # Handle null/missing values
    account_type = str(account_type).lower() if account_type and not pd.isna(account_type) else ''
    workpaper_record = str(workpaper_record).lower() if workpaper_record and not pd.isna(workpaper_record) else ''
    
    # Initialize probability from LLM
    probability = initial_probability

    # Define high-risk terminology for additional probability
    high_risk_terms = [
        'temporary funding',
        'special purpose',
        'investment holdings',
        'consulting fees',
        'distribution account',
        'clearing account',
        'special drawing',
        'development loan',
        'development fund',
        'interim funding',
        'personal transaction'
    ]
    
    # Check for existing Division 7A documentation
    if any(term in workpaper_record for term in ['div 7a', 'division 7a']):
        return 1.0, "Div 7A reconciliation worksheet already attached", False
    
    # Adjust probability based on account type and balance
    if account_type == 'liabilities':
        if balance > 0:  # Debited liability
            probability = min(probability + 0.4, 1)  # Increase probability of Division 7A
            
            # Set status based on final probability after adjustment
            if probability >= 0.5:
                status = "Liability account with debit balance - high probability of Division 7A implication"
            else:
                status = "Liability account with debit balance - low probability of Division 7A implication"
        else:  # Credited liability
            probability = min(probability, 0)  # no div 7A
            status = "Liability account with credit balance - not likely to be a Division 7A loan"    
    elif account_type == 'assets':
        # Check for high-risk terms in assets
        account_lower = str(row.get('AccountName', '')).lower()
        if any(term in account_lower for term in high_risk_terms):
            probability = min(probability + 0.3, 1)  # Increase probability for high-risk terms
            
        if probability >= 0.5:
            status = "Potential Division 7A loan"
        else:
            status = "Not likely to be a Division 7A loan"
    else:
        status = "Account type unclear - review needed"
    
    # Determine if warning is needed based on probability
    needs_warning = probability >= 0.5  # Warning for anything with 50% or higher probability
    
    return probability, status, needs_warning

def analyze_account(loan_app, row):
    """Modified account analysis to use probability"""
    result = {
        'Account': row['AccountName'],
        'Balance': row['Balance'],
        'SourceAccountType': row['SourceAccountType'],
        'Loan Detection': {
            'Probability': None,
            'Reasoning': None
        },
        'Final_Probability': None,
        'Summary': None,
        'Div7A Status': None,
        'Requires_Warning': False
    }
    
    # Get initial probability from LLM
    loan_detection = loan_app.detect_loan(row['AccountName'])
    result['Loan Detection']['Probability'] = float(loan_detection.get('probability', 0.0))
    result['Loan Detection']['Reasoning'] = loan_detection['reasoning']
    
    # Check Division 7A status with probability
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

def main():
    """Main function to run the Streamlit web application"""
    # Set up the web application title
    st.title("Div 7A Loan Assessment Tool")
    
    # Create sidebar for configuration options
    st.sidebar.header("Configuration")
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # Add file upload widget for trial balance
    trial_balance_file = st.sidebar.file_uploader("Upload Trial Balance", type=['xlsx'])
    
    # Initialize results list
    results = []
    
    # When "Analyze Accounts" button is clicked and all requirements are met
    if st.sidebar.button("Analyse Accounts") and trial_balance_file and openai_api_key:
        # Initialize loan assessment application
        loan_app = LoanAssessmentApp(openai_api_key)
        # Read Excel file into pandas DataFrame
        trial_balance_df = pd.read_excel(trial_balance_file)
        
        # Verify all required columns exist in the uploaded file
        required_columns = ['AccountName', 'Balance', 'SourceAccountType']
        missing_columns = [col for col in required_columns if col not in trial_balance_df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in trial balance: {', '.join(missing_columns)}")
            return
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Process each account in the trial balance
        for index, row in trial_balance_df.iterrows():
            # Update progress bar
            progress_bar.progress(int((index + 1) / len(trial_balance_df) * 100))
            # Analyze account and store results
            account_result = analyze_account(loan_app, row)
            results.append(account_result)
    
    # Display results if any accounts were analyzed
    if results:
        results_df = pd.DataFrame(results)
        st.header("Loan Assessment Results")
        
        # Define function to prioritize results for display
        def get_sort_priority(result):
            """
            Determines the sort priority for loan assessment results.
            Returns:
            0 - Warning (⚠️) - Requires attention, high probability
            1 - Info (ℹ️) - Medium probability
            2 - Checkmark (✅) - Low probability, no concerns
            """
            if result['Requires_Warning']:
                return 0  # ⚠️ Warning accounts first
            elif 0.5 < result['Loan Detection']['Probability'] < 0.8:
                return 1  # ℹ️ Medium probability accounts second
            else:
                return 2  # ✅ Low probability accounts last
        
        # Sort results by priority
        sorted_results = sorted(results, key=get_sort_priority)

        # The display section in main() remains the same, but with updated emoji logic:
        for result in sorted_results:
            # Determine display style based on result type
            if result['Requires_Warning']:
                status_color = 'red'
                status_emoji = "⚠️"  # Warning emoji for high risk/attention required
            elif 0.5 < result['Loan Detection']['Probability'] < 0.8:
                status_color = 'orange'
                status_emoji = "ℹ️"  # Info emoji for medium probability
            else:
                status_color = 'green'
                status_emoji = "✅"  # Green checkmark for low probability

            # Create expandable section for each account
            with st.expander(f"{status_emoji} {result['Account']} - {result['Summary']}", expanded=False):
                st.markdown(f"**Summary:** <font color='{status_color}'>{result['Summary']}</font>", unsafe_allow_html=True)
        
                st.subheader("Loan Detection")
                st.write(f"**Probability:** {result['Loan Detection']['Probability']:.1%}")
                st.write(f"**Reasoning:** {result['Loan Detection']['Reasoning']}")
                
                if result['Div7A Status']:
                    st.subheader("Div 7A Status")
                    st.write(result['Div7A Status'])

        # Add download button for full results
        sorted_results_df = pd.DataFrame(sorted_results)
        st.download_button(
            label="Download Full Results",
            data=sorted_results_df.to_csv(index=False),
            file_name="loan_assessment_results.csv",
            mime="text/csv"
        )
            
    else:
        st.write("")

# Run the application if this file is run directly
if __name__ == "__main__":
    main()