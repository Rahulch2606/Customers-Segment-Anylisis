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

# Save the trained model and scaler
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
