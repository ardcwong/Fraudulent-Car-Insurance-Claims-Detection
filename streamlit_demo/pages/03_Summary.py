import streamlit as st

st.title("Clustering Summary")
st.write("Here you can provide a short summary and recommendations")


st.header("Top Customers")
st.image("streamlit_demo/images/top.png")
st.text("Top 3 categories: grocery pos, kids and pets, gas and transport")
st.write("Since these are highly active customers who purchase frequently, we can see relatively consistent spending patterns throughout the year. From the pattern, we can observe that the first few months (Jan-Feb) of the year have the lowest amount spent.")
st.write("We can Introduce special bonuses, such as extra cashback or double points during these months for the top categories to incentivize continued spending during the holiday season.")
st.write("Since these are valuable customers we could also develop loyalty programs and create a sense of exclusivity to make customers feel special and valued. Offer exclusive perks and benefits reserved for program members, such as VIP events or early access to sales.")
st.write("To further improve the performance of the top categories, we could also explore opportunities for cross-selling and upselling. Encourage customers to explore related products or services that complement their purchases, thereby increasing transaction value.")


st.header("Engaged Customers")
st.image("streamlit_demo/images/engaged.png")
st.text("Top 3 categories: grocery pos, kids and pets, shopping pos")
st.write("Compared to the top customers, the engaged customers have volatile spending patterns. To boost some of the dips in some months for specific categories, design targeted promotions and offers that encourage higher spending, such as spend-and-get offers where customers receive a reward after spending a certain amount within a certain period.")
st.write("There are also some months where all categories need improvement we could offer seasonal incentives during key months, such as February and September to boost spending. These could include limited-time discounts or holiday-themed rewards.")

st.header("At-risk Customers")
st.image("streamlit_demo/images/risk.png")
st.text("Top 3 categories: shopping net, misc net, shopping pos")
st.write("As observed in the trend, there are only spikes in certain months, since these customers spend infrequently and less, focusing on offers specifically designed to entice at-risk customers back into using their AAC credit cards. Offers include attractive discounts or waived fees.")
st.write("For re-engagement, we could also develop targeted communication campaigns and strategies to keep members informed about program updates, rewards, and special promotions. Utilize email, SMS, push notifications, and social media to engage with members regularly.")


st.title("Supervised Learning: Linear Regression")
st.write("The purpose of creating this linear regression model was to determine the relationship between total spending and the top three spending categories: grocery_pos, kids_pets, and gas_transport. This model can also be used to create an alert system when there is a deviation between predicted and actual values.")
st.write("To achieve this, we used transaction data from January 2020 to November 2021. We grouped the total spending and total transactions for these categories from January to December 2020 to train the model. The model then predicted the total amount spent from January to November 2021.")
st.write("Note: The limitation of this model is that it was trained using only one year of data.")
st.image("streamlit_demo/images/reg.png")
st.write("Our model achieved an R-squared of 0.47, indicating a moderate positive correlation. This means that fluctuations in spending within the top three categories (grocery_pos, kids_pets, gas_transport) are likely to affect the overall spending and, consequently, the company's revenue.")
st.write("For instance, a decrease in spending or transaction volume in these categories suggests a potential drop in total revenue, and vice versa.")
st.write("This insight can help us develop an alert system to monitor trends in these critical categories. If a decreasing trend is detected, prompt action can be taken to mitigate potential revenue losses.")



