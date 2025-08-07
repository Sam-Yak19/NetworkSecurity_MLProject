### Network Security Project build using ML for Phising data

🛡️ Network Security Anomaly Detection ML Project
This project implements an end-to-end Machine Learning pipeline for detecting anomalies in network security data. It's designed to be robust, reproducible, and easy to use, featuring data ingestion, validation, transformation, model training, and a user-friendly prediction API with a web interface.

✨ Features
Data Ingestion: Securely fetches network data from a MongoDB database.

Data Validation: Ensures data quality and consistency, checking for schema adherence and missing values.

Data Transformation: Preprocesses raw data, including handling missing values using KNNImputer, to prepare it for machine learning models.

Model Training: Trains and evaluates various classification models (Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, AdaBoost) to identify the best-performing model.

MLflow Tracking: Integrates with MLflow (via Dagshub) to log experiment parameters, metrics, and models, ensuring full reproducibility and easy comparison of different runs.

Prediction API (FastAPI): Provides a high-performance, asynchronous API endpoint for real-time predictions on new network data.

User-Friendly Frontend: A simple web interface for users to upload CSV files and receive immediate predictions, complete with a sample data download option.

Containerization (Docker): Includes a Dockerfile for easy packaging and deployment of the application.

🛠️ Technologies Used
Python: Core programming language.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations, especially with transformed data arrays.

Scikit-learn: For machine learning algorithms (imputation, classification models).

FastAPI: For building the high-performance web API.

Uvicorn: An ASGI server to run the FastAPI application.

MLflow: For MLOps lifecycle management (experiment tracking, model logging).

Dagshub: A platform used for remote MLflow tracking and Git integration.

PyMongo & Certifi: For connecting to MongoDB.

python-dotenv: For managing environment variables securely.

Dill: For serializing and deserializing Python objects (models, preprocessors).

PyYAML: For working with YAML configuration files.

Jinja2Templates: For rendering HTML responses in FastAPI.

Tailwind CSS: For styling the frontend web application.

Docker: For containerizing the application.

📂 Project Structure
NetworkSystems/
├── .github/
├── .gitignore
├── .env                  # Environment variables (e.g., MONGODB_URL_KEY)
├── app.py                # FastAPI backend application
├── main.py               # Main script to run the ML pipeline (for training)
├── Dockerfile            # Docker configuration for containerization
├── requirements.txt      # Python dependencies
├── setup.py
├── README.md             # This file
├── final_model/          # Stores the trained NetworkModel (preprocessor + model)
│   └── trained_model.pkl # Example: combined model after training
├── prediction_output/    # Stores output CSVs from predictions
├── networksecurity/      # Core Python package
│   ├── __init__.py
│   ├── constant/         # Constants for pipeline configurations
│   ├── entity/           # Data models for artifacts and configurations
│   ├── exception/        # Custom exception handling
│   ├── logging/          # Custom logging setup
│   ├── Pipelines/        # ML pipeline stages (e.g., training_pipeline.py)
│   ├── components/       # Individual pipeline components (data_ingestion, data_validation, etc.)
│   └── utils/            # Utility functions (load/save objects, numpy arrays, metrics)
├── notebooks/            # Jupyter notebooks for exploration/development
├── static/               # Frontend static files (HTML, CSS, JS)
│   ├── index.html        # Main frontend application
│   └── sample_network_data.csv # Sample CSV for user download
└── templates/            # Jinja2 templates for FastAPI (e.g., table.html)
    └── table.html

🚀 Setup and Installation
Follow these steps to get the project up and running on your local machine.

Prerequisites
Python 3.8+: Ensure you have a compatible Python version installed.

Git: For cloning the repository.

1. Clone the Repository
git clone https://github.com/Sam-Yak19/NetworkSecurity_MLProject.git
cd NetworkSecurity_MLProject

2. Create and Activate Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

Windows:

python -m venv venv
.\venv\Scripts\activate

macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Once your virtual environment is activated, install all required packages:

pip install -r requirements.txt

4. Configure Environment Variables
This project uses python-dotenv to load environment variables from a .env file.

Create a file named .env in the root directory of your project (same level as app.py).

Add your MongoDB connection string to this file:

MONGODB_URL_KEY="mongodb+srv://your_username:your_password@your_cluster.mongodb.net/your_database?retryWrites=true&w=majority"

Replace your_username, your_password, your_cluster, and your_database with your actual MongoDB Atlas (or local MongoDB) connection details.

🏃 Usage
1. Training the Machine Learning Model
To train the model and save the trained NetworkModel (preprocessor + model) artifact, run the main.py script:

python main.py

This will execute the entire training pipeline:

Data Ingestion from MongoDB.

Data Validation.

Data Transformation (including KNN imputation).

Model Training and evaluation.

The best model will be saved to the final_model/ directory (e.g., final_model/trained_model.pkl).

MLflow metrics and artifacts will be logged to your Dagshub repository.

2. Running the Prediction API (FastAPI Backend)
To start the web server that hosts your prediction API:

uvicorn app:app --reload

The --reload flag is useful during development as it automatically restarts the server when code changes are detected.

The API will be available at http://127.0.0.1:8000.

3. Using the Frontend Application
Once the FastAPI backend is running:

Open your web browser and navigate to http://127.0.0.1:8000/.

You will see the Network Security Prediction web interface.

To make a prediction:

Download Sample CSV: If you don't have your own data, click the "Download Sample CSV" button to get a sample_network_data.csv file. This file demonstrates the expected input format.

Upload CSV: Click the "Click to upload" area or drag and drop your CSV file (either your own or the downloaded sample).

Get Prediction: Click the "Get Prediction" button.

The application will send your CSV to the backend, process it, and display the prediction results (including a new predicted_column) directly on the web page as an HTML table. The predicted data will also be saved as prediction_output/output.csv.

📊 MLflow Tracking
This project is configured to track experiments using MLflow, integrated with Dagshub.

Access Dagshub: Go to your Dagshub repository: https://dagshub.com/Sam-Yak19/NetworkSecurity_MLProject

Navigate to MLflow: On the Dagshub repository page, find the "MLflow" tab or section.

Explore Runs: Here, you can see all your training runs, their logged parameters, metrics (F1-Score, Precision, Recall for both training and test sets), and the saved model artifacts. This helps you compare different model versions and understand their performance.

🐳 Deployment
The project includes a Dockerfile for easy containerization, which is the first step towards deploying your application to cloud platforms.

To build the Docker image:

docker build -t network-security-app .

After building, you can run the container:

docker run -p 8000:8000 network-security-app

This will run your FastAPI application inside a Docker container, accessible on port 8000 of your host machine. For production deployment, consider platforms like Google Cloud Run, AWS Elastic Beanstalk, or Azure App Service.

🐛 Troubleshooting
RuntimeError: Form data requires "python-multipart" to be installed.:

Solution: Activate your virtual environment and run pip install python-multipart.

OSError: Cannot save file into a non-existent directory: 'prediction_output':

Solution: The app.py code should automatically create this directory. If not, ensure you're using the latest app.py version or manually create the prediction_output folder in your project root.

The file final_model/trained_model.pkl does not exists:

Solution: This means the training pipeline (python main.py) has not been run successfully, or the path in app.py to load the model is incorrect. Run python main.py first, and verify the trained_network_model_path in app.py matches where main.py saves the model.

mlflow.exceptions.RestException: INTERNAL_ERROR: Response: {'error': 'unsupported endpoint, please contact support@dagshub.com'}:

Solution: This was addressed by modifying ModelTrainer.track_ml_flow to use mlflow.sklearn.save_model followed by mlflow.log_artifacts instead of direct mlflow.sklearn.log_model. Ensure you have the latest ModelTrainer code.

👋 Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Make your changes.

Commit your changes (git commit -m 'Add new feature').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
