import streamlit_authenticator as stauth

hashed_pw = stauth.Hasher(['your_password_here']).generate()
print(hashed_pw[0])
