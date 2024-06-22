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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

#%%
df_en = pd.read_csv('actual_adopcio_en.csv') 
print()

df_en = df_en.drop(columns=['Link'])
print(df_en.dtypes)

#%% Kreiranje kolone sa starosnim podacima u mesecima
df_en['Age_in_Months'] = df_en['Age_Years'] * 12 + df_en['Age_Months']

# Provera da li kolone sadrže numeričke vrednosti
print(df_en[['Age_Years', 'Age_Months', 'Age_in_Months']].dtypes)

# Vizualizacija starosne strukture životinja
plt.figure(figsize=(10, 6))
plt.hist(df_en['Age_in_Months'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribucija starosti životinja za usvajanje')
plt.xlabel('Starost u mesecima')
plt.ylabel('Broj životinja')
plt.grid(True)
plt.show()
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


#%% Izdvajanje i brojanje rasa
breed_counts = df_en['Description'].value_counts()

# Vizualizacija popularnosti rasa
plt.figure(figsize=(12, 8))
breed_counts.plot(kind='bar', color='skyblue')
plt.title('Popularnost rasa životinja za usvajanje')
plt.xlabel('Rasa životinje')
plt.ylabel('Broj životinja')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Analiza najčešće usvajanih rasa
top_breeds = breed_counts.head(10)
print("Najčešće usvajane rase:")
print(top_breeds)
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


#%% Vizualizacija sezonskih trendova
#Pretvaranje kolone Entry_Date u datetime format ako nije već pretvorena
df_en['Entry_Date'] = pd.to_datetime(df_en['Entry_Date'], format='%Y-%m-%d')

# Dodavanje kolone sa mesecom
df_en['Month'] = df_en['Entry_Date'].dt.month
df_en['Year'] = df_en['Entry_Date'].dt.year

plt.figure(figsize=(10, 6))
df_en['Month'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Sezonski trend broja usvojenih životinja')
plt.xlabel('Mesec')
plt.ylabel('Broj usvojenih životinja')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

#%% 
# Grupisanje po godinama i mesecima
yearly_counts = df_en.groupby(['Year', df_en['Entry_Date'].dt.month])['Entry_Date'].count().unstack()

# Provera koje su sve godine prisutne u podacima
years_present = sorted(df_en['Year'].unique())  # Sortiranje godina od najmanje do najveće

# Vizualizacija sezonskih trendova za svaku godinu posebno
colors = plt.cm.get_cmap('tab10', len(yearly_counts.columns))

for year in years_present:
    if year in yearly_counts.index:
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(yearly_counts.columns):
            plt.plot(yearly_counts.columns, yearly_counts.loc[year], marker='o', color=colors(i), label=f'Month {col}')

        plt.title(f'Sezonski trend broja usvojenih životinja za godinu {year}')
        plt.xlabel('Mesec')
        plt.ylabel('Broj usvojenih životinja')
        plt.xticks(yearly_counts.columns)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Godina {year} nema podataka za prikaz.")
        
        
# %% NEURONSKA MREZA

# Pretvaranje kategoričkih podataka u numeričke
df_en['Gender'] = pd.factorize(df_en['Gender'])[0]
df_en['Size'] = pd.factorize(df_en['Size'])[0]

# Odabir ulaznih i ciljnih podataka
X = df_en[['Age_in_Months', 'Gender', 'Size', 'Month']].values
y = df_en['Month'].values  # Ciljni podatak za demonstraciju

# Podela podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacija podataka
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Izlazni sloj za predviđanje broja usvojenih životinja
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

model.summary()


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)


loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss}")
print(f"Test MAE: {mae}")

# Prikazivanje grafika gubitka tokom treninga
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()