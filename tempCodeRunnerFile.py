import pandas as pd

# Load the transaction data
df = pd.read_csv("transactions.csv")

# Define rules for suspicious transactions
def is_suspicious(row):
    return row['Amount'] > 30000 or row['Merchant'] in ['Unknown', 'Hackermarket']

# Apply the rules
df['Suspicious'] = df.apply(is_suspicious, axis=1)

# Filter suspicious ones
suspicious_df = df[df['Suspicious'] == True]

# Save suspicious transactions to a new file
suspicious_df.to_csv("suspicious_transactions.csv", index=False)

print("Suspicious transactions saved to suspicious_transactions.csv")
