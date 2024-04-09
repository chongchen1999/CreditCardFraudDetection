import pandas as pd
# Load the dataset
data_path = r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\datasets\creditcard_2023.csv'
data = pd.read_csv(data_path)

# Assuming 'data' is your DataFrame and 'Class' is your target column

# Count the number of occurrences of each class
class_counts = data['Class'].value_counts()

# Calculate the total number of samples
total_samples = len(data)

# Calculate the proportion of each class
class_proportions = class_counts / total_samples

# Output the proportions
print("number of 0:", class_counts[0])
print("number of 1:", class_counts[1])

print("Proportion of Negative Samples (Class = 0):", class_proportions[0])
print("Proportion of Positive Samples (Class = 1):", class_proportions[1])


# Calculate the average amount of all transactions
average_amount_all = data['Amount'].mean()

# Calculate the average amount of fraud transactions
average_amount_fraud = data[data['Class'] == 1]['Amount'].mean()

# Calculate the average amount of not fraud transactions
average_amount_not_fraud = data[data['Class'] == 0]['Amount'].mean()

# Output the averages
print("Average Amount of All Transactions:", average_amount_all)
print("Average Amount of Fraud Transactions:", average_amount_fraud)
print("Average Amount of Not Fraud Transactions:", average_amount_not_fraud)
