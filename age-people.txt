data = """\
Age agegroup height status yearsmarried
21 adult 6.0 single -1
2 child 3 married 0
18 adult 5.7 married 20
221 elderly 5 widowed 2
34 child -7 married 3
"""



with open("people.txt", "w") as file:
    file.write(data)

#creating the ruleset

import pandas as pd
import matplotlib.pyplot as plt

# Read data from file
df = pd.read_csv("people.txt", delim_whitespace=True)

# Define rules
def check_rules(row):
    violations = []
    
    # Rule 1
    if not (0 <= row['Age'] <= 150):
        violations.append("Age out of range")
    
    # Rule 2
    if row['Age'] <= row['yearsmarried']:
        violations.append("Age not greater than yearsmarried")
    
    # Rule 3
    if row['status'] not in ["married", "single", "widowed"]:
        violations.append("Invalid status")
    
    # Rule 4
    if row['Age'] < 18 and row['agegroup'] != "child":
        violations.append("Incorrect agegroup for age < 18")
    elif 18 <= row['Age'] <= 65 and row['agegroup'] != "adult":
        violations.append("Incorrect agegroup for age between 18 and 65")
    elif row['Age'] > 65 and row['agegroup'] != "elderly":
        violations.append("Incorrect agegroup for age > 65")
    
    return violations

# Check rules for each row and collect results
df['Violations'] = df.apply(check_rules, axis=1)

# Summarize results
summary = df[['Age', 'agegroup', 'height', 'status', 'yearsmarried', 'Violations']]
print(summary)


# Flatten the list of violations
violations_list = [violation for sublist in summary['Violations'] for violation in sublist]

# Count occurrences of each violation type
violation_counts = pd.Series(violations_list).value_counts()

# Plot the results
violation_counts.plot(kind='bar')
plt.title('Number of Violations per Rule')
plt.xlabel('Rule Violated')
plt.ylabel('Number of Violations')
plt.show()
