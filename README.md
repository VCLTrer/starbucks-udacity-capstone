# Starbucks Capstone Challenge: Offer Response Prediction

## Project Overview

This project aims to analyze customer response data from the Starbucks rewards mobile app to determine which demographic groups respond best to different types of offers. By combining transaction, demographic, and offer data, we build and evaluate a machine learning model to predict whether a customer will respond to an offer based on their demographic information and the type of offer received.

## Data Description

The dataset consists of three JSON files:

1. **portfolio.json**: Contains offer IDs and metadata about each offer (duration, type, etc.)
   - `id` (string): Offer ID
   - `offer_type` (string): Type of offer (e.g., BOGO, discount, informational)
   - `difficulty` (int): Minimum required spend to complete an offer
   - `reward` (int): Reward given for completing an offer
   - `duration` (int): Time for offer to be open, in days
   - `channels` (list of strings)

2. **profile.json**: Contains demographic data for each customer
   - `age` (int): Age of the customer
   - `became_member_on` (int): Date when customer created an app account
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
   - Create a response variable indicating whether an offer was completed.
   - Generate dummy variables for categorical features such as gender and offer type.

## Exploratory Data Analysis (EDA)

### Distribution of Income

![Income Distribution](https://via.placeholder.com/800x400.png?text=Income+Distribution)

### Distribution of Age

![Age Distribution](https://via.placeholder.com/800x400.png?text=Age+Distribution)

### Membership Duration

![Membership Duration](https://via.placeholder.com/800x400.png?text=Membership+Duration)

### Response Rate by Offer Type

![Response Rate by Offer Type](https://via.placeholder.com/800x400.png?text=Response+Rate+by+Offer+Type)

### Response Rate by Gender

![Response Rate by Gender](https://via.placeholder.com/800x400.png?text=Response+Rate+by+Gender)

## Detailed Analysis of Features

### Portfolio Data

- **id**: Unique identifier for each offer. Used for merging with transcript and profile data.
- **offer_type**: Type of offer (e.g., BOGO, discount, informational). Essential for analyzing effectiveness with different demographic groups.
- **difficulty**: Minimum required spend to complete an offer. May influence a customer's likelihood to complete an offer.
- **reward**: Reward for completing an offer. Higher rewards might drive higher response rates.
- **duration**: Time for which the offer is valid. Shorter durations might create urgency, while longer durations might allow more time for completion.
- **channels**: Distribution channels used for the offer (e.g., web, email, mobile, social). Offers sent through preferred channels might see higher engagement.

### Profile Data

- **age**: Age of the customer. Different age groups might respond differently to various offer types.
- **became_member_on**: Date when the customer created an app account. Used to calculate membership duration.
- **gender**: Gender of the customer (M, F, O). Might play a role in offer preferences and responses.
- **id**: Unique identifier for each customer. Used for merging with other datasets.
- **income**: Customer's income. Income levels might affect purchasing power and responsiveness to offers.

### Transcript Data

- **event**: Type of event (e.g., transaction, offer received, offer viewed, offer completed). Tracks customer interactions with offers.
- **person**: Customer ID. Used for merging with profile data.
- **time**: Time in hours since the start of the test. Used to analyze the timing of customer interactions.
- **offer_id**: ID of the offer related to the event. Used for merging with portfolio data.
- **amount**: Transaction amount. Relevant for analyzing spending patterns.

### Abnormalities and Characteristics

1. **Age Outliers**: Address outliers in the `age` feature, such as implausibly high values.
2. **Missing Values**: Handle missing values in the dataset, particularly in the `gender` feature.
3. **Gender Representation**: Consider the imbalance in the `gender` feature.
4. **Income Distribution**: Address wide range and potential outliers in the `income` feature.
5. **Event Timing**: Convert `time` from hours to days for better interpretability.
6. **Offer View vs. Completion**: Analyze the relationship between offers received, viewed, and completed.

## Model Development

### Train-Test Split

Split the data into training and testing sets to evaluate model performance on unseen data.

### Grid Search with Cross-Validation

Use `GridSearchCV` to perform hyperparameter optimization for the `RandomForestClassifier`.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")
```

### Model Training and Evaluation

Train the `RandomForestClassifier` with the best parameters identified from the grid search. Evaluate the model using various metrics, including accuracy, precision, recall, F1 score, and a confusion matrix.

### Confusion Matrix

![Confusion Matrix](https://via.placeholder.com/800x400.png?text=Confusion+Matrix)

## Rationale for Metric Selection

### Accuracy

Measures the proportion of correctly classified instances out of the total instances. While straightforward, it can be misleading in imbalanced datasets.

### Precision

Measures the proportion of true positive predictions out of all positive predictions. Important in marketing to avoid targeting non-responsive customers.

### Recall

Measures the proportion of true positive predictions out of all actual positives. Ensures that the model captures most of the actual responses, maximizing campaign effectiveness.

### F1 Score

Harmonic mean of precision and recall. Provides a balanced metric that considers both false positives and false negatives, useful in imbalanced datasets.

### Confusion Matrix

Provides a detailed breakdown of true positives, false positives, true negatives, and false negatives, helping understand the types of errors the model makes.

## Algorithms and Techniques

### RandomForestClassifier

**RandomForestClassifier** is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) of the individual trees.

- **Principle**: Constructs a 'forest' of decision trees, each trained on a random subset of data and features. Final prediction by averaging the predictions (regression) or majority voting (classification).
- **Assumptions**: Assumes a group of weak learners (shallow trees) can form a strong learner. Averaging multiple trees reduces variance without increasing bias.
- **Parameters**: 
  - `n_estimators`: Number of trees in the forest.
  - `max_features`: Number of features to consider for the best split.
  - `max_depth`: Maximum depth of each tree.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
  - `bootstrap`: Whether bootstrap samples are used when building trees.

### GridSearchCV

**GridSearchCV** is used for hyperparameter optimization by exhaustively searching over a specified parameter grid.

- **Principle**: Performs a search over specified parameter values for an estimator using

 cross-validation to evaluate performance.
- **Assumptions**: Assumes that the best model can be found by exhaustively searching over all parameter combinations. Cross-validation provides a reliable estimate of model performance.
- **Parameters**: 
  - `param_grid`: Dictionary with parameter names as keys and lists of parameter settings to try.
  - `cv`: Cross-validation splitting strategy.
  - `n_jobs`: Number of jobs to run in parallel.
  - `verbose`: Controls the verbosity.

## Challenges and Insights

### Challenge: Data Leakage

Dealing with data leakage was crucial. Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance metrics. We carefully re-examined data splitting and preprocessing steps to ensure no overlap between training and test sets.

### Insight: Importance of Feature Engineering

Feature engineering played a critical role. Creating meaningful features, such as membership duration and dummy variables for categorical features, significantly impacted model accuracy. This step underscores the importance of domain knowledge in transforming raw data into valuable features for machine learning models.

## Future Work

### Improvement: Model Optimization

Experimenting with other algorithms and advanced hyperparameter tuning techniques could further improve model performance. Techniques like Bayesian optimization or ensemble methods might offer better results.

### Improvement: Additional Features

Exploring additional features such as customer engagement metrics (e.g., frequency of app usage, past offer completions) could provide more insights and improve prediction accuracy. Combining external data sources, such as social media activity or geographic information, could also enhance the model.

## How to Run the Project

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Install the required dependencies using `pip install -r requirements.txt`.
3. **Run Notebooks**: Open and run the Jupyter notebooks in the `notebooks/` directory to explore the data and build the model.
4. **Run Scripts**: Execute the Python scripts in the `scripts/` directory for data preprocessing and model training.

## Authors

- Thato Rammoko

## Acknowledgments

- Starbucks for providing the data.
- Udacity for the project guidance and support.
