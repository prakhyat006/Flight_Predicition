# Flight Price Prediction 🛫

A machine learning project that predicts flight prices based on various flight characteristics using Python and scikit-learn.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project develops a predictive model to estimate flight prices in the Indian aviation market. The system analyzes historical flight data to predict ticket prices based on factors like airline, route, departure time, duration, and number of stops.

## 📊 Dataset

The project uses flight booking data with the following features:
- **Airline**: Flight carrier (Air India, IndiGo, Jet Airways, etc.)
- **Date_of_Journey**: Travel date
- **Source**: Departure city
- **Destination**: Arrival city
- **Route**: Flight path with stops
- **Dep_Time**: Departure time
- **Arrival_Time**: Arrival time
- **Duration**: Total flight duration
- **Total_Stops**: Number of intermediate stops
- **Additional_Info**: Extra flight information
- **Price**: Target variable (flight cost in INR)

## ✨ Features

### Data Preprocessing
- **Missing Value Handling**: Removes incomplete records
- **Feature Engineering**: 
  - Extracts day and month from journey dates
  - Separates time components (hours/minutes) from departure and arrival times
  - Converts duration from text format to numerical minutes
- **Categorical Encoding**: Transforms text categories to numerical values
- **Feature Selection**: Removes complex features for model simplicity

### Model Implementation
- **Multiple Algorithm Comparison**: Linear Regression, Decision Tree, Random Forest
- **Comprehensive Evaluation**: R², RMSE, and MAE metrics
- **Visualization**: Performance comparison charts
- **Prediction Interface**: Easy-to-use prediction system for new flight data

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/flight-price-prediction.git
cd flight-price-prediction
```

2. **Install required packages**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

3. **For Jupyter Notebook**
```bash
pip install jupyter
```

## 🚀 Usage

### Running the Notebook
1. Open the Jupyter notebook:
```bash
jupyter notebook flight_predict2.ipynb
```

2. Upload your dataset (`Data_Train.xlsx`) when prompted

3. Run all cells to:
   - Load and preprocess the data
   - Train multiple models
   - Compare model performance
   - Make predictions on new data

### Making Predictions
```python
# Example prediction input
manual_input = {
    'Airline': 'Air India',
    'Source': 'Mumbai',
    'Destination': 'Hyderabad',
    'Total_Stops': 'non-stop',
    'Journey_day': 20,
    'Journey_month': 6,
    'Dep_hour': 14,
    'Dep_min': 30,
    'Arrival_hour': 16,
    'Arrival_min': 15,
    'Duration': 105  # in minutes
}
```

## 📈 Model Performance

The project compares three machine learning algorithms:

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | - | - | - |
| Decision Tree | - | - | - |
| Random Forest | - | - | - |

*Note: Actual performance metrics will be displayed after running the notebook*

## 📁 Project Structure

```
flight-price-prediction/
│
├── flight_predict2.ipynb    # Main Jupyter notebook
├── Data_Train.xlsx          # Training dataset (add your file)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## 🔧 Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning library
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **openpyxl** - Excel file handling

## 📊 Results

### Key Insights
- Flight prices vary significantly based on airline choice
- Departure time and duration are important price factors
- Number of stops affects ticket pricing
- Seasonal patterns influence flight costs

### Model Comparison
The Random Forest model typically shows the best performance due to its ability to:
- Handle non-linear relationships
- Capture feature interactions
- Reduce overfitting through ensemble averaging

## 🚀 Future Improvements

- [ ] **Cross-validation** for more robust model evaluation
- [ ] **Hyperparameter tuning** using GridSearchCV or RandomizedSearchCV
- [ ] **Feature importance analysis** to understand key price drivers
- [ ] **One-hot encoding** for categorical variables
- [ ] **Time series analysis** for seasonal price patterns
- [ ] **Web scraping** for real-time data collection
- [ ] **Web application** deployment using Flask/Streamlit
- [ ] **Advanced models** like XGBoost or Neural Networks

## 📝 Requirements

Create a `requirements.txt` file with:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
jupyter>=1.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [prakhyat006](https://github.com/prakhyat006)

## 🙏 Acknowledgments

- Dataset source: https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh
- Inspiration from various flight price prediction studies


---

⭐ **Star this repository if you found it helpful!**
