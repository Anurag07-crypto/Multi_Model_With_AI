import streamlit as st  
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzJk0aYfhcrI8tODgvWBNQ9af1iitzmoZ5PQ&s")
st.title("Welcome to this MULTI_MODE_WITH_A")
st.write("Enter your name")
name = st.text_input(" ",placeholder="Jhon Doe")
find = st.selectbox("Where you find this model",["LinkedIn","Other source","Friend Referal"])
st.title("Now Select The Model To Use")
st.set_page_config(
    page_title="Multi_Mode_With_A",
    page_icon="https://cdn-icons-png.flaticon.com/512/8167/8167135.png" 
)
col_1, col_2, col_3, col_4 = st.columns(4)
pg = st.navigation(
    [
        st.Page("page\page1.py", title="Chat With Bot", icon="🏠"),
        st.Page("page\page2.py", title="Leaf Classifier", icon="⚙️"),
        st.Page("page\page3.py", title="Real vs Ai image Identifier", icon="📊"),
        st.Page("page\page4.py", title="Indian Bird Classifier", icon="📊"),
    ]
)
pg.run()
with col_1:
    st.image("https://img.freepik.com/premium-vector/chat-bot-concept-illustration_114360-5522.jpg")
    if st.button("Chat with Bot"):
        st.switch_page("page\page1.py")
    st.text("This model is currently not working good")
with col_2:
    st.image("https://www.datocms-assets.com/117510/1722393357-act_native_leaf_diy_classifcation_simple_leaf_shapes.jpg")
    if st.button("Leaf classifier"):
        st.switch_page("page\page2.py")
with col_3:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQK9-5Jz7JeFvV21D6Hxyj0Q7HsNaYJEOLEGQ&s")
    if st.button("Real vs Ai image identifier"):
        st.switch_page("page\page3.py")
with col_4:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREF8dHvPsKqd7F6u8dDjQPEUHZIsMsU-2n2A&s")
    if st.button("Indian bird classifier"):
        st.switch_page("page\page4.py")



