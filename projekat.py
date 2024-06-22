# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:50:10 2024

@author: Korisnik
"""

#%% Biblioteke
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from googletrans import Translator
from googletrans.models import Translated
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from datetime import datetime


#%% Citanje podataka
df = pd.read_csv('actual_adopcio.csv') 
print(df.head(5))
print(df.dtypes)
#%% Prevodjenje dataset-a i imena kolona:
    
translator = Translator()

# Funkcija za prevođenje kolone sa rukovanjem izuzetaka
def translate_column(column, src_lang, dest_lang):
    translated_data = []
    for text in column:
        if pd.isna(text):
            translated_data.append(text)
        else:
            try:
                translation = translator.translate(text, src=src_lang, dest=dest_lang)
                translated_data.append(translation.text)
            except Exception as e:
                translated_data.append(f"Error: {e}")
    return translated_data

def translate_dangerous_column(df, column_name='Dangerous'):
    df[column_name] = df[column_name].astype(str).str.strip().str.upper()
    translation_dict = {'S': 'Yes', 'N': 'No'}
    df[column_name] = df[column_name].map(translation_dict)
    return df

# Prevođenje naziva kolone 
df.rename(columns = {
    'ANY_REF': 'Reference_Year',
    'REFERENCIA': 'Reference',
    'EDAT_ANYS': 'Age_Years',
    'EDAT_MESOS': 'Age_Months',
    'DIA_ENTRADA': 'Entry_Date',
    'CODI_RACA': 'Breed_Code',
    'DESCRIPCIO': 'Description',
    'PERILLOS': 'Dangerous',
    'SEXE': 'Gender',
    'CAPA': 'Color',
    'TAMANY': 'Size',
    'NOM_ANIMAL': 'Animal_Name',
    'OBSERVACIONS_WEB': 'Observations_Web',
    'LINK': 'Link'}, inplace = True)

# Prevođenje sadržaja kolone 
#df['Description'] = translate_column(df['Description'], src_lang='es', dest_lang='en')
#df['Observations_Web'] = translate_column(df['Observations_Web'], src_lang='es', dest_lang='en')
df['Description'] = translate_column(df['Description'], src_lang='ca', dest_lang='en')
df['Observations_Web'] = translate_column(df['Observations_Web'], src_lang='ca', dest_lang='en')
df['Color'] = translate_column(df['Color'], src_lang='ca', dest_lang='en')
df = translate_dangerous_column(df)

# Parsiranje datuma
df['Entry_Date'] = pd.to_datetime(df['Entry_Date'], format='%Y-%m-%d')

# Priprema podataka
df['Entry_Year'] = df['Entry_Date'].dt.year
df['Entry_Month'] = df['Entry_Date'].dt.month
df['Entry_Day'] = df['Entry_Date'].dt.day
#df.drop('Entry_Date', axis=1, inplace=True)

# Objedinjavanje dve kolone
df['Age_Years_Total'] = df['Age_Years'] + df['Age_Months'] / 12

print(df.head(5))
print(df.dtypes)
# Čuvanje prevedenog dataset-a
df.to_csv('C:\\Users\\Korisnik\\Desktop\\anja\\MITNOP 2\\MITNOP_projekat\\actual_adopcio_en.csv', index=False)

#%% Fuzzy logika

# Definisanje membership funkcija
def young_membership(age):
    if age <= 5:
        return 1
    elif 5 < age < 8:
        return (8 - age) / 3
    else:
        return 0

def middle_membership(age):
    if 3 <= age <= 7:
        return (age - 3) / 4
    elif 7 < age <= 10:
        return (10 - age) / 3
    else:
        return 0

def old_membership(age):
    if age >= 15:
        return 1
    elif 10 < age < 15:
        return (age - 10) / 5
    else:
        return 0

def low_dangerousness(age_membership):
    return max(age_membership['young'], 0.3 * age_membership['middle'])

def medium_dangerousness(age_membership):
    return max(0.5 * age_membership['middle'], 0.7 * age_membership['old'])

def high_dangerousness(age_membership):
    return max(0.3 * age_membership['middle'], age_membership['old'])

def calculate_fuzzy_dangerousness(age):
    age_membership = {
        'young': young_membership(age),
        'middle': middle_membership(age),
        'old': old_membership(age)
    }
    low = low_dangerousness(age_membership)
    medium = medium_dangerousness(age_membership)
    high = high_dangerousness(age_membership)

    # Defuzzifikacija
    dangerousness = (low * 0.2 + medium * 0.5 + high * 0.8) / (low + medium + high)
    return dangerousness

df['Fuzzy_Dangerousness'] = df['Age_Years_Total'].apply(calculate_fuzzy_dangerousness)

print(df[['Age_Years_Total', 'Fuzzy_Dangerousness']].head())

#%% Vizualizacija rezultata



plt.figure(figsize=(12, 6))
plt.scatter(df['Age_Years_Total'], df['Fuzzy_Dangerousness'], alpha=0.5)
plt.xlabel('Age in Years')
plt.ylabel('Fuzzy Dangerousness')
plt.title('Fuzzy Dangerousness vs Age')
plt.grid(True)
plt.show()


#%% Model za predikciju Reference Year-a

# Priprema podataka
X = df[['Age_Years_Total', 'Fuzzy_Dangerousness']]
y = df['Reference_Year']

# Deljenje podataka na treniranje i testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treniranje Random Forest modela
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Treniranje Logistic Regression modela
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

# Upoređivanje tačnosti modela
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Logistic Regression Accuracy: {lr_accuracy}")

df['Entry_Date'] = pd.to_datetime(df['Entry_Date'], format='%Y-%m-%d')

#%% Priprema podataka
df['Entry_Year'] = df['Entry_Date'].dt.year
df['Entry_Month'] = df['Entry_Date'].dt.month
df['Entry_Day'] = df['Entry_Date'].dt.day

X = df[['Age_Years_Total', 'Fuzzy_Dangerousness', 'Entry_Year']]
y = df['Reference_Year']

# Deljenje podataka na treniranje i testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treniranje Random Forest modela
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Treniranje Logistic Regression modela
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

# Upoređivanje tačnosti modela
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Logistic Regression Accuracy: {lr_accuracy}")
