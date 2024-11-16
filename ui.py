import streamlit as st
import requests

# Streamlit UI setup
st.title("Query Processing App")
st.write("Enter your query below, and we'll process it to provide results.")

# Input query
query = st.text_input("Enter your Case query:")

# Submit button
if st.button("Submit"):
    if query.strip():
        # Show a loading spinner
        with st.spinner("Processing your query..."):
            # FastAPI endpoint
            api_url = "http://127.0.0.1:8000/process_query/"  # Replace with your FastAPI server URL if different
            payload = {"case_details": query, "case_id":"123abc","case_title":"xyz"}
            headers = {
                "Content-Type": "application/json",
            }

            try:
                # Send POST request to FastAPI
                response = requests.post(api_url, json=payload, headers=headers)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Parse and display the results
                result = response.json()  # Assuming the API returns JSON
                st.success("Query processed successfully!")
                
                if 'response' in result:
                    st.write("Case Result:")
                    st.write(result['response'])  # Display the embedding result
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query before submitting.")


