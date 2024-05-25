import streamlit as st

st.title("Customer Demographics")
st.write("Based on AAC Credit Card Transactions as of December 7, 2021")


st.header("Gender")
st.text("AAC Credit Card Holders demographic shows a strong gender disparity with males")
st.text("comprising 94% and females just 6% of the cardholders.")
st.image("streamlit_demo/images/Gender.png")

st.header("Active Customers per year")
st.image("streamlit_demo/images/Active.png")
st.text("Compared to 78 users in 2020, there was an increase in active customers in 2021,")
st.text("with 88 active users.")

st.header("Age")
st.image("streamlit_demo/images/Age.png")
st.text("The age range of cardholders is between 51 and 90, with the mean age being 67.")

st.header("Generation Distribution")
st.image("streamlit_demo/images/Generation.png")
st.text("The majority of these customers are from the Baby Boomer generation.")



st.title("Transaction and Spending")
st.text("Looking at these YoY Monthly Transaction counts and amounts, we can see that despite")
st.text("transaction counts remaining completely on trend, it's evident that certain months")
st.text("on monthly transaction amount struggle to be consistent with the overall trend.")

st.header("YoY Monthly Transaction Counts")
st.image("streamlit_demo/images/mon_trans.png")
st.header("YoY Monthly Transaction Amount")
st.image("streamlit_demo/images/mon_trans_amt.png")
st.text("Note: There was a significant dip in December 2021, primarily due to")
st.text("incomplete transaction records ending on December 7th.")


st.title("Category Spending")
st.text("In here, we can see the top categories that drive AAC customers' spending.")

st.image("streamlit_demo/images/cat.png")
st.text("1 | The high expenditure on groceries and kids and pets indicates that the cardholders")
st.text("    prioritize essential and family-related expenses.")
st.text("2 | Significant spending on gas_transport, food_dining, and shopping points to a")
st.text("    lifestyle that values mobility and social activities.")
st.text("")
st.text("Overall, these insights suggest that AAC Credit Card Holders are family-oriented, mobile,")
st.text("socially active, and financially confident, providing a clear direction for targeted")
st.text("marketing and service offerings.")















#st.header("1. Top customer spending behavior by category")
#st.image("streamlit_demo/images/total_spending.png")
#st.text("Travel is the top spender's highest spending, followed by onsite grocery and onsite shopping.")

