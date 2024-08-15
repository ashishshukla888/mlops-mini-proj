from mlflow.tracking import MlflowClient
from flask import Flask, render_template ,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle
import os 

# from browser to secret auth
# mlflow.set_tracking_uri('https://dagshub.com/ashishshukla888/mlops-mini-proj.mlflow')
# dagshub.init(repo_owner='ashishshukla888', repo_name='mlops-mini-proj', mlflow=True)

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ashishshukla888"
repo_name = "mlops-mini-proj"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

app = Flask(__name__)

# load model from model registry
def get_latest_model_version(model_name):
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)

    # show
    return render_template('index.html', result= result[0])

app.run(debug=True,host="0.0.0.0")



# app = Flask(__name__) 



# app = Flask(__name__)

# # load model from model registry
# def get_latest_model_version(model_name):
#     client = MlflowClient()
#     latest_version = client.get_latest_versions(model_name, stages=["Production"])
#     if not latest_version:
#         latest_version = client.get_latest_versions(model_name, stages=["None"])
#     return latest_version[0].version if latest_version else None

# model_name = "my_model"
# model_version = get_latest_model_version(model_name)

# model_uri = f'models:/{model_name}/{model_version}'
# model = mlflow.pyfunc.load_model(model_uri)

# vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))



# @app.route('/')
# def home():
#     return render_template('index.html',result=None)

# @app.route('/predict', methods=['POST'])
# def predict():

#     text = request.form['text']

#     # clean
#     text = normalize_text(text)

#     # bow
#     features = vectorizer.transform([text])

#     # prediction
#     result = model.predict(features)

#     # show
#     return render_template('index.html', result= result[0])

# app.run(debug=True)



# # load model from model registry
# def get_latest_model_version(model_name):
#     client = MlflowClient()
#     latest_version = client.get_latest_versions(model_name, stages=["Production"])
#     if not latest_version:
#         latest_version = client.get_latest_versions(model_name, stages=["None"])
#     return latest_version[0].version if latest_version else None

# # 1.> load model

# model_name = "my_model"
# model_version = get_latest_model_version(model_name)

# model_uri = f'models:/{model_name}/{model_version}'
# model = mlflow.pyfunc.load_model(model_uri)

# vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))



# @app.route('/')
# def home():
#     return render_template('index.html')



# @app.route('/predict',methods=['POST'])
# def predict():
#     text = request.form['text']

#     # clean
#     text = normalize_text(text)

#     # bow
#     features = vectorizer.transform([text])

#     # prediction
#     result = model.predict(features)

#     return render_template('index.html', result= result[0])
# app.run(debug=True)