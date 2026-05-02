<<<<<<< HEAD
CICIDS Multi-Class Intrusion Detection System
A machine learning-based web application that detects and classifies network traffic into multiple attack categories using the CICIDS 2017 dataset. The system is built using Random Forest and provides real-time predictions through an interactive Streamlit interface.

Project Overview
This project focuses on identifying malicious network activity by analyzing traffic flow features. It classifies traffic into multiple categories such as:
* BENIGN
* DoS
* DDoS
* PortScan
* BruteForce
* Bot
* Infiltration

The goal is to build a practical intrusion detection system that can assist in identifying cyber threats effectively.

Approach--
The project follows a structured machine learning pipeline:

1. Data Collection

   * CICIDS 2017 dataset (network flow data)

2. Data Preprocessing

   * Merging multiple CSV files
   * Removing duplicates and missing values
   * Grouping similar attack types
   * Handling data imbalance

3. Feature Engineering

   * Selecting top 35 important features using feature importance

4. Model Training

   * Random Forest Classifier
   * Train-test split with stratification
   * Standard scaling for normalization

5. Model Evaluation

   * Accuracy and classification report
   * Performance analysis across attack classes

6. Deployment (Local)

   * Streamlit-based UI for real-time predictions


Dataset
* Dataset Used: CICIDS 2017
* Type:** Network flow-based dataset
* Total Features: ~80+
* Selected Features: Top 35 (based on importance)

The dataset includes both normal traffic and multiple attack scenarios captured in realistic network environments.



Project Structure

CICIDS-MULTICLASS-IDS/
│
├── app/
│   └── app.py                 # Streamlit UI
│
├── src/
│   ├── train.py              # Model training pipeline
│   ├── preprocessing.py      # Data preprocessing
│   └── utils.py              # Helper functions
│
├── models/
│   ├── final_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── selected_features.pkl
│
├── data/
│   └── raw/                  # Original dataset files
│
├── requirements.txt
└── README.md


Installation & Setup

1. Clone the Repository
git clone https://github.com/Prathibha459/CICIDS-MultiClass-IDS-Detection.git
cd CICIDS-MultiClass-IDS-Detection

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install Dependencies
pip install -r requirements.txt


4. Train the Model (if needed)
python src/train.py


5. Run the Application
streamlit run app/app.py

Open in browser:
http://localhost:8501


Usage

* **Manual Input:** Enter feature values directly in the UI
* **CSV Upload:** Upload a single-row CSV with required features
* The system outputs:

  * Predicted attack type
  * Confidence score
  * Probability distribution across classes


Model Details

* **Algorithm:** Random Forest Classifier
* **Features Used:** Top 35 important features
* **Scaling:** StandardScaler
* **Label Encoding:** LabelEncoder


Limitations

* Imbalanced classes like *Infiltration* have very few samples
* Model performance depends on input quality
* Uses flow-based features, not real-time packet capture


Future Improvements

* Use advanced models like XGBoost or LightGBM
* Apply SMOTE for better handling of rare classes
* Real-time traffic integration
* Model explainability (SHAP / feature insights)


Conclusion

This project demonstrates how machine learning can be applied to cybersecurity for detecting and classifying network intrusions. It provides a practical and interactive system for understanding attack patterns and improving network security.


=======
# ML-Powered-Cloud-Based-Security-System
>>>>>>> 36625c217ae6f70588799bd30eaed770c9e721af
