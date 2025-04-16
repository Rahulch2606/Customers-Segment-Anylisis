from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('kmeans_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Mapping dictionary for cluster names
cluster_names = {
    0: "High Income, High Spending",
    1: "Medium Income, Medium Spending",
    2: "Low Income, High Spending",
    3: "Low Income, Low Spending",
    4: "High Income, Low Spending"
}

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')
# Define a route for clustering new data
@app.route('/cluster', methods=['POST'])
def cluster_data():
    # Get the JSON data from the request
    json_data = request.json

    # Extract input data (assuming JSON structure like {"income": 50000, "spending": 1000})
    income = json_data.get('income')
    spending = json_data.get('spending')

    # Validate input data
    if income is None or spending is None:
        return jsonify({'error': 'Income and spending values are required.'}), 400

    # Convert input data to DataFrame
    new_data = pd.DataFrame({'income': [income], 'spending': [spending]})

    # Scale the new data using the loaded scaler
    scaled_new_data = scaler.transform(new_data)

    # Predict clusters for the new data
    new_clusters = model.predict(scaled_new_data)

    # Get cluster names
    cluster_name = cluster_names.get(new_clusters[0])

    # Convert new_clusters[0] to int to ensure JSON serializability
    cluster = int(new_clusters[0])

    # Return the predicted cluster label and name as JSON response
    return jsonify({'cluster': cluster, 'cluster_name': cluster_name})

if __name__ == '__main__':
    app.run(debug=True)
