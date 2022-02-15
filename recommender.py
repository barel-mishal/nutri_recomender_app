import os
from sklearn import datasets
import streamlit as st
import pandas as pd

from recommender_system import content_base_recommender

__DIRNAME__ = os.path.dirname(os.path.realpath(__file__))
DATASET = pd.read_csv(os.path.join(__DIRNAME__, 'csvs', 'israeli_data.csv'))

def main():
  # taking only the items names 
  data = DATASET[['shmmitzrach']].rename(columns={'shmmitzrach': 'שם מצרך'})
  # write into the app
  st.write('שמות מצרכי המזון במידה', data)
  
  item = st.text_input('כדי לרשום שם של מוצר מזון כדי לקבל המלצה למשל חסה: ')
  if item != '':
    recommend(item)
  return 


def recommend(item):
  r1, r2, r3 = content_base_recommender(item)
  r1
  r2
  r3

if __name__ == '__main__':
  main()