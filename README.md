# Starbucks Capstone Challenge: Offer Response Prediction

## Project Overview

This project analyzes customer response data from the Starbucks rewards mobile app to determine which demographic groups respond best to different offers. By combining transaction, demographic, and offer data, we build and evaluate a machine learning model to predict whether a customer will respond to an offer based on their demographic information and the type of offer received.

## Project Reflections
Visit the link below to find my reflections

https://medium.com/@thatorammoko/predicting-customer-responses-to-starbucks-offers-4980f6a27770

## Data Description

The dataset consists of three JSON files:

1. **portfolio.json**: Contains offer IDs and metadata about each offer (duration, type, etc.)
   - `id` (string): Offer ID
   - `offer_type` (string): Type of offer (e.g., BOGO, discount, informational)
   - `difficulty` (int): Minimum required spend to complete an offer
   - `reward` (int): Reward given for completing an offer
   - `duration` (int): Time for offer to be open, in days
   - `channels` (list of strings): Channels where the offer was distributed (e.g., web, email, mobile, social)

2. **profile.json**: Contains demographic data for each customer
   - `age` (int): Age of the customer
   - `became_member_on` (int): Date when the customer created an app account
   - `gender` (str): Gender of the customer (M, F, O)
   - `id` (str): Customer ID
   - `income` (float): Customer's income

3. **transcript.json**: Contains records for transactions, offers received, offers viewed, and offers completed
   - `event` (str): Record description (e.g., transaction, offer received, offer viewed, offer completed)
   - `person` (str): Customer ID
   - `time` (int): Time in hours since the start of the test
   - `value` (dict): Either an offer ID or transaction amount depending on the record

## Project Structure

- `data/`: Contains the raw JSON data files
- `notebooks/`: Contains Jupyter notebooks for data exploration and model development
- `scripts/`: Contains Python scripts for data preprocessing and model training
- `results/`: Contains results such as model evaluation metrics and visualizations
- `README.md`: This ReadMe file

## Data Preprocessing

1. **Load Data**: Load the datasets from JSON files.
2. **Expand 'value' Column**: Expand the 'value' column in the `transcript` data to get transaction amounts and offer IDs.
3. **Filter Events**: Filter out the relevant events: 'offer received', 'offer viewed', and 'offer completed'.
4. **Merge Data**: Merge the offer events with the customer profiles and portfolio data.
5. **Feature Engineering**:
   - Calculate the membership duration.
   - Create a response variable indicating whether the offer was completed or not.
   - Create dummy variables for categorical features such as gender and offer type.

## Model Development

1. **Data Splitting**: Split the data into training and testing sets.
2. **Grid Search with Cross-Validation**: Use `GridSearchCV` to find the best hyperparameters for the `RandomForestClassifier`.
3. **Model Training**: Train the model using the best parameters from the grid search.
4. **Model Evaluation**: Evaluate the model's performance using accuracy, precision, recall, F1 score, and confusion matrix metrics.

## Results

- **Best Parameters**: The best parameters found using grid search.
- **Evaluation Metrics**: The performance metrics of the model on both the training and testing sets.
- **Confusion Matrix**: A visual representation of the model's performance.

## Visualizations

- **Response Rates by Income Group and Offer Type**: A bar plot showing the response rates for different income groups and offer types.
- **Response Rates by Age and Offer Type**: A line plot showing the response rates for different age groups and offer types.
- **Response Rates by Gender and Offer Type**: A bar plot showing the response rates for different genders and offer types.

## How to Run the Project

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Install the required dependencies using `pip install -r requirements.txt`.
3. **Run Notebooks**: Open and run the Jupyter notebooks in the `notebooks/` directory to explore the data and build the model.
4. **Run Scripts**: Execute the Python scripts in the `scripts/` directory for data preprocessing and model training.

## Conclusion

This project demonstrates how to analyze and predict customer responses to different types of offers using demographic and transaction data. By leveraging machine learning techniques and cross-validation, we can identify which demographic groups are most responsive to specific offer types, providing valuable insights for targeted marketing strategies.

## Authors

- Thato Rammoko

## Acknowledgments

- Starbucks for providing the data
- Udacity for the project guidance and support

---

Feel free to customize this ReadMe file based on your specific project details and preferences.
