
import streamlit as st
import pickle

# Load trained model and TF-IDF vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Page configuration
st.set_page_config(
    page_title="Smart Spam Mail Assistant",
    page_icon="📧"
)

st.title("📧 Smart Spam Mail Assistant")

st.write(
    "Enter an email message or upload a text file "
    "to check whether it is Spam or Ham."
)

# Email input
email_text = st.text_area(
    "Paste your email here:",
    height=180,
    placeholder="Enter your email message..."
)

# File upload
uploaded_file = st.file_uploader(
    "Or upload a .txt file",
    type=["txt"]
)

if uploaded_file is not None:
    email_text = uploaded_file.read().decode(
        "utf-8", errors="replace"
    )
    st.info("Text file uploaded successfully!")

# Prediction button
if st.button("Predict", type="primary"):

    if email_text.strip():

        # Convert email into numerical features
        input_features = vectorizer.transform([email_text])

        # Model prediction
        prediction = model.predict(input_features)

        st.divider()

        if prediction[0] == 0:

            st.error("🚨 Spam Mail Detected!")

            st.subheader("💬 Message from your Email Assistant")

            st.write("""
            Hello! I've checked this email, and it appears suspicious.

            It may contain misleading information or an unsafe link.

            Please be careful!

            - Don't click on unknown links.
            - Don't share your passwords or banking details.
            - Avoid replying to suspicious emails.

            My suggestion: Move this email to your spam folder.
            """)

        else:

            st.success("✅ Genuine (Ham) Mail Detected!")

            st.subheader("💬 Message from your Email Assistant")

            st.write("""
            Hello! Good news!

            Your email appears to be genuin.

            You can read the message, but always stay alert.

            - Check the sender's email address.
            - Be careful with unexpected attachments.
            - Never share sensitive information unnecessarily.

            Have a great day!
            """)

    else:
        st.warning("Please enter or upload an email first.")
