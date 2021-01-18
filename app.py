from flask import Flask, render_template, request
import joblib
import numpy as np


directory = "model/"
pca_path = directory + "sklearn_decomposition__pca_PCA.pkl"
model_path = directory + "catboost_core_CatBoostClassifier.pkl"

pca = joblib.load(pca_path)
classifier = joblib.load(model_path)


app = Flask(__name__)


def get_input_data(request):
    texture_mean = float(request.form['Texture mean'])
    area_mean = float(request.form['Area mean'])  
    smoothness_mean = float(request.form['Smoothness mean'])
    concave_points_mean = float(request.form['Concave Points mean'])
    symmetry_mean = float(request.form['Symmetry mean'])
    fractal_dimension_mean = float(request.form['Fractal Dimension mean'])
    texture_se = float(request.form['Texture standard error'])
    area_se = float(request.form['Area standard error'])
    smoothness_se = float(request.form['Smoothness standard error'])
    compactness_se = float(request.form['Compactness standard error'])
    concavity_se = float(request.form['Concavity standard error'])
    concave_points_se = float(request.form['Concave Points standard error'])
    symmetry_se = float(request.form['Symmetry standard error'])
    fractal_dimension_se = float(request.form['Fractal Dimension standard error'])
    concavity_worst = float(request.form['Concavity worst'])
    symmetry_worst = float(request.form['Symmetry worst'])
    fractal_dimension_worst = float(request.form['Fractal Dimension worst'])

    input_data = np.array([[texture_mean, area_mean, smoothness_mean, concave_points_mean,
                       symmetry_mean, fractal_dimension_mean, texture_se, area_se, smoothness_se, 
                       compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, 
                       concavity_worst, symmetry_worst, fractal_dimension_worst]])

    return input_data


def calculate_prediction(input_data):
    input_data_reduced = pca.transform(input_data)
    prediction = classifier.predict(input_data_reduced)
    return prediction


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/diagnose')
def diagnose():
    return render_template('diagnose.html')


@app.route('/diagnose/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = get_input_data(request)
        prediction = calculate_prediction(input_data)
        return render_template('predict.html', prediction=prediction)


    

if __name__ == '__main__':
    app.run(debug=True)