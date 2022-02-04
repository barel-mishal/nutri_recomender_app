import os
import streamlit as st
import pandas as pd

__DIRNAME__ = os.path.dirname(os.path.realpath(__file__))
dataset = pd.read_csv(os.path.join(__DIRNAME__, 'csvs', 'israeli_data.csv'))

st.write('שמות מצרכי המזון במידה', dataset['shmmitzrach'])


item = st.text_input('כדי לרשום שם של מוצר מזון כדי לקבל המלצה למשל חסה: ')

st.write(item)

