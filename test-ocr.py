import base64
import io
import json
import pandas as pd
from PIL import Image
import ollama
import streamlit as st
import re
def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encodes an image file to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_ocr_output_from_image(image_base64: str, model: str = "x/llama3.2-vision:11b") -> dict:
    """Sends an image to the Llama OCR model and returns structured text output as JSON."""
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": """Act as an OCR Expert. You will be provided with an image containing details 
            either about products or raw materials, or invoices. Your task is to accurately extract all 
            relevant information from the image, ensuring that each data point is paired with the
            correct column header. Extract all columns like (raw materials, inventory, re order, purchases, availability, to order). 
            Structure the extracted information in a well-formatted JSON format. I just want the json, no extra text or statements in the response.

            """,
            "images": [image_base64]
        }]
    )
    
    raw_text = response.get('message', {}).get('content', '').strip()
    return extract_json_from_text(raw_text)

def convert_json_to_dataframe(json_data: dict) -> pd.DataFrame:
    """Converts JSON data into a structured DataFrame."""
    if isinstance(json_data, list):  # If JSON is already a list of dictionaries
        df = pd.DataFrame(json_data)
    elif isinstance(json_data, dict):  # Convert a single dictionary to a DataFrame
        df = pd.DataFrame([json_data])
    else:
        df = pd.DataFrame()
    return df

def extract_json_from_text(text: str) -> dict:
    """Extracts and parses JSON from mixed response formats."""
    print('aya')
    print(text)
    try:
        # Direct JSON case
        return json.loads(text)
    except json.JSONDecodeError:
        # Handle cases where JSON is wrapped in extra text
        json_matches = re.findall(r'\{.*\}|\[.*\]', text, re.DOTALL)
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return {}  # Return empty dict if no valid JSON is found


def main():
    st.title("OCR Data Extractor")
    st.write("Upload an image containing product, raw material, or invoice details, and extract structured JSON data.")

    uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Encode the image to base64
        st.write("Processing the image...")
        base64_image = encode_image_to_base64(image)

        # Get OCR output
        try:
            json_data = get_ocr_output_from_image(base64_image)
            #print(json_data)
            st.write("**Extracted JSON Data:**")
            st.json(json_data)

            # Convert JSON to DataFrame
            df = convert_json_to_dataframe(json_data)

            if not df.empty:
                # Display DataFrame
                st.write("**Structured Data:**")
                st.dataframe(df)

                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="ocr_extracted_data.csv",
                    mime="text/csv"
                )

                # Download Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='OCR Data')
                output.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=output,
                    file_name="ocr_extracted_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.write("No structured data extracted.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()