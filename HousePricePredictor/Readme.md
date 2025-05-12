
# California Housing Price Prediction

## Project Overview
This is an end-to-end machine learning project for predicting California housing prices using a pre-trained model.

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation Steps
1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Ensure you have the pickled model (`regmodel.pkl`) in the project directory

5. Run the Flask application
```bash
python app.py
```

### Input Features
The model expects 8 input features:
1. Median Income
2. House Age
3. Average Rooms
4. Average Bedrooms
5. Population
6. Average Occupancy
7. Latitude
8. Longitude

### Prediction Output
The prediction is in 100k USD increments. For example, a prediction of 3.5 means $350,000.

## Model Training
The model was trained on the California Housing dataset using scikit-learn, with feature preprocessing including:
- Imputation
- Scaling
- Polynomial Feature Generation
- Feature Selection

## Technologies Used
- Backend: Flask
- Frontend: React
- ML Libraries: scikit-learn, XGBoost
- Styling: Tailwind CSS

## Notes
- Ensure all input features are numeric
- The model provides an estimated housing price based on the given features
