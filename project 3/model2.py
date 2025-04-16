import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the data
data = pd.read_csv("customers.csv")

# Select features
features = data[["income", "spending"]]

# Feature Scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Train the KMeans model
model = KMeans(n_clusters=5, n_init=10, random_state=1)
clusters = model.fit_predict(scaled_features)

# Save the trained model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Function to predict clusters for new data
def predict_clusters(new_data):
    """
    Predicts the cluster labels for new data.
    
    Args:
        new_data (array-like): New data to be predicted, where each row represents a sample and each column represents a feature.
    
    Returns:
        array-like: Predicted cluster labels for the new data.
    """
    # Scale the input features
    scaled_new_data = scaler.transform(new_data)
    
    # Predict cluster labels
    predicted_clusters = model.predict(scaled_new_data)
    
    return predicted_clusters

# Example usage
new_data = [[50, 60], [70, 30], [20, 80]]  # Example new data
predicted_labels = predict_clusters(new_data)
print("Predicted Cluster Labels:", predicted_labels)
