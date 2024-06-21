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
df = translate_dangerous_column(df)

# Objedinjavanje dve kolone
df['Age_Years_Total'] = df['Age_Years'] + df['Age_Months'] / 12

print(df.head(5))
print(df.dtypes)
# Čuvanje prevedenog dataset-a
df.to_csv('C:\\Users\\Korisnik\\Desktop\\anja\\MITNOP 2\\MITNOP_projekat\\actual_adopcio_en2.csv', index=False)

#%% 