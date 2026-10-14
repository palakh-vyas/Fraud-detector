<<<<<<< HEAD
import mysql.connector
import pandas as pd

# 1. Load the transaction data
df = pd.read_csv("transactions.csv")


# 2. Define rules for suspicious transactions
def is_suspicious(row):
  return row["Amount"] > 30000 or row["Merchant"] in ["Unknown", "Hackermarket"]


# Apply rules and save filtered output
df["Suspicious"] = df.apply(is_suspicious, axis=1)
suspicious_df = df[df["Suspicious"] == True]
suspicious_df.to_csv("suspicious_transactions.csv", index=False)
print("Suspicious transactions saved to suspicious_transactions.csv")

# 3. Connect to MySQL and insert all records
try:
  conn = mysql.connector.connect(
      host="localhost",
      user="root",
      password="MaTaJ@090",  # Replace with your MySQL root password
      database="fraud_detection_db",
  )
  cursor = conn.cursor()

  # Create table if it doesn't exist yet
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            Amount FLOAT,
            Merchant VARCHAR(255),
            Suspicious BOOLEAN
        )
    """)

  # Insert data row by row
  for _, row in df.iterrows():
    cursor.execute(
        """
            INSERT INTO transactions (Amount, Merchant, Suspicious) 
            VALUES (%s, %s, %s)
        """,
        (float(row["Amount"]), str(row["Merchant"]), bool(row["Suspicious"])),
    )

  conn.commit()
  cursor.close()
  conn.close()
  print("Data successfully uploaded and synced with MySQL database!")

except Exception as e:
  print(f"MySQL Error: {e}")
=======
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
>>>>>>> dfd47746834b7d5b7dbd2ab4e929d7494bf3836e
