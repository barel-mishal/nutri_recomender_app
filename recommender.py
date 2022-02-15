import os
import streamlit as st
import pandas as pd

from recommender_system import content_base_recommender

__DIRNAME__ = os.path.dirname(os.path.realpath(__file__))
dataset = pd.read_csv(os.path.join(__DIRNAME__, 'csvs', 'israeli_data.csv'))

st.write('שמות מצרכי המזון במידה', dataset[['shmmitzrach']].rename(columns={'shmmitzrach': 'שם מצרך'}))


item = st.text_input('כדי לרשום שם של מוצר מזון כדי לקבל המלצה למשל חסה: ')
if item != '':
  r1, r2, r3 = content_base_recommender(item)
  r1
  r2
  r3

