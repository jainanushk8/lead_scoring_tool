import streamlit_authenticator as stauth

passwords = ['yourpassword1', 'yourpassword2']
hashed_passwords = [stauth.Hasher().hash(pw) for pw in passwords]

print(hashed_passwords)
