import streamlit_authenticator as stauth

# Your password here
password = "Your Password Here"

# Generate bcrypt hash
hash_pw = stauth.Hasher.hash(password)
print("bcrypt hash:", hash_pw)