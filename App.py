import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

#Title
st.title("🛒 E-commerce Recommendation System")

#Load Data
df = pd.read_csv("E-Commerce cleaned data.csv")

#Prepare basket
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'] \
           .sum().unstack().fillna(0)

#Convert to boolean
basket = basket > 0

#Apply Apriori Algorithm
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)

#Product list
product_list = df['Description'].unique()

#User input
product = st.selectbox("Select Product:", product_list)

#Recommendation function
def recommend(product_name):
    result = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    return result[['consequents', 'confidence']]

#Show recommendations
if st.button("Get Recommendations"):
    result = recommend(product)

    if result.empty:
        st.write("No recommendations found.")
    else:
        for i in result['consequents']:
            st.write(list(i))
