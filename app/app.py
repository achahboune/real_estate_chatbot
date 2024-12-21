from flask import Flask, render_template, request, jsonify
import pandas as pd
import google.generativeai as genai
import json

app = Flask(__name__)

# Configure the Gemini API
genai.configure(api_key='YOUR_API_KEY')
model = genai.GenerativeModel('models/gemini-1.5-flash')

# Load the dataset
df_read = pd.read_csv('real_estate_data.csv')

# Function to extract JSON from the response text
def extract_json(response_text):
    start_marker = "```json\n"
    end_marker = "\n```"
    start = response_text.find(start_marker)
    if start != -1:
        start += len(start_marker)
        end = response_text.find(end_marker, start)
        if end != -1:
            json_content = response_text[start:end]
        else:
            raise ValueError("End marker not found in the response.")
    else:
        json_content = response_text.strip()
    return json_content

# Function to apply filter to the dataset
def apply_filter(df, filter_dict):
    temp = df.copy()
    for key, value in filter_dict.items():
        if key in df.columns:
            if key != 'quality':
                temp = temp[temp[key] == value]
            elif key == 'quality':
                temp = temp[temp['quality'].isin(value)]
    return temp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_query = data['query']
        
        # Prompt engineering to extract filter and prompt type
        prompt1 = f"""
        Task: Analyze the real estate query and extract filtering conditions.
        
        Available columns and their types:
        - bedrooms (int): Number of bedrooms
        - price (int): Price in dollars
        - state (str): State name
        - quality (int): Quality rating (1-10)
        - size_in_meters (int): Size in square meters
        - real_estate_number (str): Unique identifier
        
        User query: "{user_query}"
        Extract the available columns data and return them as a JSON file 
        and a "prompt_type": One of ["quality", "price", "general"] 
        Example response:
            {{
                "bedrooms": 2,
                "price": [300000],
                "state": "Toulkana",
                "prompt_type": "price",
            }};
        
        Provide only the JSON response, no additional text.
        """
        response = model.generate_content(prompt1)
        filter_json = extract_json(response.text)
        filtered_docs = json.loads(filter_json)
        
        # Apply the filter to the dataset
        filtered_df = apply_filter(df_read, filtered_docs)
        
        # Generate the report based on prompt_type
        if not filtered_df.empty:
            if filtered_docs['prompt_type'] == 'quality':
                prompt = f"""
                Real Estate Quality Analysis Task
                    Original user query: "{user_query}"
                    Filtered data: "{filtered_df.to_html()}"
                    Give the user a nice and short report of the quality from the retrieved document in a very professional way.
                """
            elif filtered_docs['prompt_type'] == 'price':
                prompt = f"""
                Real Estate Price Analysis Task
                    Original user query: "{user_query}"
                    Filtered data: "{filtered_df.to_html()}"
                    Give the user a nice and short report of the price from the retrieved document in a very professional way.
                """
            elif filtered_docs['prompt_type'] == 'general':
                prompt = f"""
                Real Estate General Analysis Task
                    Original user query: "{user_query}"
                    Filtered data: "{filtered_df.to_html()}"
                    Give the user a nice and short report of the requirements of the user from the retrieved document in a very professional way.
                """
            else:
                return jsonify({'response': 'Unknown prompt type.'})
            
            response = model.generate_content(prompt)
            report = response.text.strip()
            return jsonify({'response': report})
        else:
            return jsonify({'response': 'No matching records found.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)