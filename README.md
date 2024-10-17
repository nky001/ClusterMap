
# ClusterMap - Customer Segmentation using Machine Learning

**ClusterMap** is a machine learning-based customer segmentation tool designed to group customers based on provided data. It uses clustering algorithms to identify customer groups and offers insightful statistical analysis for each cluster, visualized through interactive charts and graphs. This project can be used for customer behavior analysis, marketing strategies, and more.

## Features

- **Upload CSV/Excel Files:** Upload customer data for segmentation.
- **Feature Mapping:** Map your CSV/Excel columns to model features.
- **Clustering:** K-Means clustering model is used to segment customers.
- **Cluster Statistics:** Get detailed statistics for each cluster, such as customer count and mean values for features.
- **Interactive Charts:** Visualize clusters and their statistics with interactive, customizable charts.
- **Real-time Predictions:** Instant predictions based on uploaded customer data.

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Data Visualization:** Matplotlib, JavaScript
- **Frontend:** HTML, CSS, JavaScript (Charts.js)
- **Deployment:** Can be easily deployed on a Flask web server or any other platform.

## Model Training

The machine learning model is trained using the **K-Means clustering algorithm**. Here's a quick overview of the training process:

1. **StandardScaler for Feature Scaling:**  
   Before training the model, the data is scaled using `StandardScaler` from Scikit-learn. This ensures that all features contribute equally to the clustering process by normalizing the data to have zero mean and unit variance.

2. **K-Means Clustering:**  
   The **KMeans** algorithm from Scikit-learn is used to cluster customers into different groups based on the selected features. The model looks for natural groupings in the data by minimizing the variance within clusters.
   
   The number of clusters (K) can be configured based on the dataset and project requirements.

   ```python
   from sklearn.cluster import KMeans
   from sklearn.preprocessing import StandardScaler
   
   # Scaling the features
   scaler = StandardScaler()
   scaled_data = scaler.fit_transform(data[selected_features])

   # Training the KMeans model
   kmeans = KMeans(n_clusters=4, random_state=42)
   kmeans.fit(scaled_data)
   
   # Adding cluster labels to the data
   data['Cluster'] = kmeans.labels_
   ```

3. **Model Predictions:**  
   Once the model is trained, it can predict the cluster for each customer based on their features. The model's output includes the cluster labels and statistics for each cluster.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nky001/ClusterMap.git
   cd ClusterMap
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server:**

   ```bash
   flask run
   ```

5. **Access the web app:**  
   Open your browser and go to `http://127.0.0.1:5000`.

## Usage

1. **Upload Data:**  
   Upload your customer data in CSV or Excel format.
   
2. **Map Features:**  
   Select the features (columns) to be used for segmentation.

3. **Generate Clusters:**  
   The K-Means model will analyze the data and segment customers into clusters.

4. **View Statistics and Charts:**  
   Get insights for each cluster with detailed statistics and visualize the results with interactive charts.

## Model Details

- **Algorithm:** K-Means Clustering
- **Scaling:** The model uses StandardScaler for feature scaling before clustering.
- **Customization:** Easily adjust the number of clusters by changing the K value in the model.


## Example Data Format

Ensure your data file contains a `CustomerID` column along with the following features:

- `PurchaseFrequency`: The frequency of purchases by the customer.
- `TotalQuantity`: The total quantity of items purchased by the customer.
- `TotalSpend`: The total amount of money spent by the customer.
- `Recency`: The number of days since the last purchase.

### Example:

| CustomerID | PurchaseFrequency | TotalQuantity | TotalSpend | Recency |
|------------|-------------------|---------------|------------|---------|
| 1          | 5                 | 20            | 150.75     | 10      |
| 2          | 3                 | 10            | 100.50     | 20      |

Make sure the uploaded CSV or Excel file includes all these columns for accurate customer segmentation.


## Live Demo

You can view the live version of this customer segmentation project here: [clustermap.up.railway.app](https://clustermap.up.railway.app)

This demo allows you to upload your dataset, map the features, and visualize customer clusters along with detailed statistics.


## License

This project is licensed under the MIT License.

