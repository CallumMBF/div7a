import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a list of account names with varying loan-related characteristics
account_names = [
    # Explicit Loans
    "Director Loan Receivable",
    "Intercompany Loan to Subsidiary",
    "Long-Term Loan Receivable",
    "Short-Term Loan to Partner",
    
    # Potentially Ambiguous
    "Advance to Supplier",
    "Receivable from Related Party",
    "Deferred Payment Asset",
    "Credit Facility Prepayment",
    
    # Non-Loan Accounts
    "Office Equipment",
    "Prepaid Insurance",
    "Inventory Deposit",
    "Accounts Receivable - Trade",
    "Rent Prepayment",
    
    # Tricky Cases
    "Interest-Bearing Note",
    "Convertible Asset",
    "Financing Arrangement",
    "Installment Receivable"
]

# Generate random total values
total_values = np.random.randint(10000, 500000, size=len(account_names))

# Create DataFrame
trial_balance_df = pd.DataFrame({
    'Account Name': account_names,
    'Current Year Total': total_values
})

# Save to Excel
trial_balance_df.to_excel('mock_trial_balance.xlsx', index=False)

print("Mock trial balance created successfully:")
print(trial_balance_df)