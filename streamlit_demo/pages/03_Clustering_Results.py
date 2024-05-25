import streamlit as st

st.title("CUSTOMER SEGMENTATION")
st.write("Based on AAC Credit Card Transactions as of December 7, 2021")


st.header("Elbow Method and Silhouette Score for Optimal Number of Cluster")
st.image("streamlit_demo/images/cluster.png")
st.text("1 | Based on our clustering analysis , the elbow method suggests 3 optimal")
st.text("    clusters while the silhouette score indicates 5.") 
st.text("")
st.text("2 | The 3-cluster model offers a simpler segmentation of high, medium, and")
st.text("    low-value customers. While the 5-cluster model provides more detailed")
st.text("    insights, identifying 5 distinct subgroups")


st.title("MEET THE CLUSTERS")
st.image("streamlit_demo/images/Kmeans.png")
st.text("Top | Interpretation: These are highly active and valuable customers who purchase")
st.text("      frequently and spend a lot. They are your top customers.")
st.text("")
st.text("Engaged | Interpretation: Customers in this cluster are also active but with a")
st.text("          moderate level of spending compared to Top Customers.")
st.text("")
st.text("At-Risk | Interpretation: Customers in this cluster have not made a purchase in a")
st.text("          long time, buy infrequently, and spend less. They might be at risk of") 
st.text("          churning.")

st.header("Transaction and Spending Distribution of each Cluster")
st.image("streamlit_demo/images/dist.png")
st.text("Top Customers: 50-55% of transaction volume & expenditure")
st.text("Engaged Customers: Significant contribution to volume & expenditure")
st.text("At-Risk Customers: Minimal transaction activity. Only notable expenditure for 4")
st.text("months")

st.header("Customer Spending per Cluster per Category")
st.image("streamlit_demo/images/treemap.png")
st.text("Top and Engaged customers spend the most in physical grocery stores (grocery_pos).")
st.text("At-Risk customers spend the most in  online shopping (shopping_net)")
