import pandas as pd

mood = pd.read_csv("_Digital Wellbeing Audit – Screen Time vs. Productivity_Mood - Form responses 1.csv")
print("Columns:", mood.columns.tolist())
print(mood.head())