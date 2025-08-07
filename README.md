# 🛡️ Network Security ML Project

An end-to-end Machine Learning pipeline for detecting anomalies in network security data using advanced ML techniques. This project focuses on phishing data detection and is designed to be robust, reproducible, and production-ready.

## 📋 Table of Contents

- [Features](#-features)
- [Technologies Used](#️-technologies-used)
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
- [MLflow Tracking](#-mlflow-tracking)
- [Docker Deployment](#-docker-deployment)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **🔍 Data Ingestion**: Securely fetches network data from MongoDB database
- **✅ Data Validation**: Ensures data quality and consistency with schema validation
- **🔄 Data Transformation**: Advanced preprocessing with KNNImputer for missing values
- **🤖 Model Training**: Multiple ML algorithms (Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, AdaBoost)
- **📊 MLflow Integration**: Complete experiment tracking via Dagshub
- **⚡ FastAPI Backend**: High-performance, asynchronous API for real-time predictions
- **🌐 Web Interface**: User-friendly frontend for CSV upload and predictions
- **🐳 Containerization**: Docker support for easy deployment
- **📈 Model Comparison**: Automatic best model selection based on performance metrics

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.8+, Pandas, NumPy |
| **ML/AI** | Scikit-learn, MLflow, Dill |
| **Web Framework** | FastAPI, Uvicorn, Jinja2Templates |
| **Database** | MongoDB, PyMongo |
| **Frontend** | HTML, Tailwind CSS, JavaScript |
| **DevOps** | Docker, python-dotenv |
| **Tracking** | Dagshub, MLflow |

## 📂 Project Structure

```
NetworkSystems/
├── .github/                    # GitHub workflows and configurations
├── .gitignore                 # Git ignore file
├── .env                       # Environment variables (MongoDB URL)
├── app.py                     # FastAPI backend application
├── main.py                    # ML pipeline execution script
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup configuration
├── README.md                  # Project documentation
│
├── final_model/               # Trained model storage
│   └── trained_model.pkl     # Serialized model artifact
│
├── prediction_output/         # Prediction results
│   └── output.csv            # Generated predictions
│
├── networksecurity/           # Core Python package
│   ├── __init__.py
│   ├── constant/             # Configuration constants
│   ├── entity/               # Data models and configurations
│   ├── exception/            # Custom exception handling
│   ├── logging/              # Logging configuration
│   ├── Pipelines/            # ML pipeline stages
│   ├── components/           # Pipeline components
│   └── utils/                # Utility functions
│
├── notebooks/                # Jupyter notebooks for EDA
├── static/                   # Frontend static files
│   ├── index.html           # Main web interface
│   └── sample_network_data.csv # Sample data for testing
│
└── templates/                # Jinja2 templates
    └── table.html           # Results display template
```

## 🚀 Setup and Installation

### Prerequisites

- **Python 3.8+** installed on your system
- **Git** for version control
- **MongoDB** connection (local or Atlas)

### 1. Clone the Repository

```bash
git clone https://github.com/Sam-Yak19/NetworkSecurity_MLProject.git
cd NetworkSecurity_MLProject
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
MONGODB_URL_KEY="mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority"
```

> **Note:** Replace the placeholder values with your actual MongoDB connection details.

## 🏃 Usage

### 1. Train the ML Model

Execute the complete training pipeline:

```bash
python main.py
```

This process includes:
- 📥 Data ingestion from MongoDB
- ✅ Data validation and quality checks
- 🔄 Data transformation and preprocessing  
- 🤖 Model training with multiple algorithms
- 💾 Best model selection and saving
- 📊 MLflow experiment logging

### 2. Start the API Server

Launch the FastAPI backend:

```bash
uvicorn app:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

### 3. Use the Web Interface

1. **Access the Application**: Navigate to `http://127.0.0.1:8000/`
2. **Download Sample Data**: Click "Download Sample CSV" for test data
3. **Upload Your CSV**: Drag and drop or click to upload your network data
4. **Get Predictions**: Click "Get Prediction" to analyze your data
5. **View Results**: Results display in an HTML table and save to `prediction_output/output.csv`

## 📊 MLflow Tracking

Track your experiments and model performance:

1. **Access Dagshub**: Visit [https://dagshub.com/Sam-Yak19/NetworkSecurity_MLProject](https://dagshub.com/Sam-Yak19/NetworkSecurity_MLProject)
2. **Navigate to MLflow**: Click the "MLflow" tab in your repository
3. **Explore Runs**: Compare different training runs, parameters, and metrics
4. **Model Artifacts**: Access saved models and preprocessing pipelines

### Tracked Metrics
- F1-Score (Training & Test)
- Precision (Training & Test)  
- Recall (Training & Test)
- Model Parameters
- Data Transformation Steps

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t network-security-app .
```

### Run Container

```bash
docker run -p 8000:8000 network-security-app
```

The application will be accessible at `http://localhost:8000`

### Production Deployment Options
- **Google Cloud Run**
- **AWS Elastic Beanstalk**
- **Azure App Service**
- **Heroku**

## 📖 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

### Key Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | Web interface homepage |
| `/predict` | POST | Upload CSV and get predictions |
| `/download-sample` | GET | Download sample CSV file |

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. Form data requires "python-multipart"**
```bash
pip install python-multipart
```

**2. Cannot save file into non-existent directory**
- Ensure the `prediction_output/` directory exists
- The application should create it automatically

**3. Model file does not exist**
```bash
python main.py  # Train the model first
```

**4. MLflow connection errors**
- Check your Dagshub credentials
- Verify network connectivity
- Ensure MLflow tracking URI is correct

**5. MongoDB connection issues**
- Verify your `.env` file configuration
- Check MongoDB Atlas whitelist settings
- Test connection string separately

## 📈 Model Performance

The system evaluates multiple algorithms and automatically selects the best performer:

- **Random Forest**: Ensemble method for robust predictions
- **Decision Tree**: Interpretable tree-based model  
- **Gradient Boosting**: Sequential weak learner improvement
- **Logistic Regression**: Linear probabilistic classifier
- **AdaBoost**: Adaptive boosting ensemble

## 🔒 Security Considerations

- Environment variables for sensitive data
- Input validation for uploaded files
- Secure MongoDB connections
- Docker containerization for isolation


## 🙏 Acknowledgments

- **MLflow** for experiment tracking
- **Dagshub** for MLOps platform
- **FastAPI** for the high-performance web framework
- **Scikit-learn** for machine learning algorithms

---

<div align="center">

**[⭐ Star this repository](https://github.com/Sam-Yak19/NetworkSecurity_MLProject)** if you find it helpful!

Made with ❤️ for the cybersecurity community

</div>
