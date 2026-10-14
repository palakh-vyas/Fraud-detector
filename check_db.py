import mysql.connector

try:
  # Connect to your MySQL database
  conn = mysql.connector.connect(
      host="localhost", user="root", password="MaTaJ@090", database="fraud_detection_db"
  )
  cursor = conn.cursor()

  # Query all rows from the transactions table
  cursor.execute("SELECT * FROM transactions LIMIT 5;")
  rows = cursor.fetchall()

  print("\n--- First 5 Rows in MySQL Database ---")
  for row in rows:
    print(row)

  # Check total count of records
  cursor.execute("SELECT COUNT(*) FROM transactions;")
  total_count = cursor.fetchone()[0]
  print(f"\nTotal transactions stored in database: {total_count}")

  # Check how many are flagged as suspicious
  cursor.execute("SELECT COUNT(*) FROM transactions WHERE Suspicious = 1;")
  suspicious_count = cursor.fetchone()[0]
  print(f"Total flagged suspicious transactions: {suspicious_count}\n")

  cursor.close()
  conn.close()

except Exception as e:
  print(f"Error connecting to database: {e}")