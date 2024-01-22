from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import plotly.express as px
import joblib

data=pd.read_csv('data/heart.csv')
data=data.rename(columns = {
    'age':'Age', 
    'trestbps':'RestingBP', 
    'chol':'Cholesterol', 
    'thalach':'MaxHR', 
    'oldpeak':'Oldpeak', 
    'fbs':'FastingBS',
    'sex':'Sex',
    'cp':'ChestPainType',
    'restecg':'RestingECG',
    'exang':'ExerciseAngina',
    'slope':'ST_Slope',
    'target':'HeartDisease'})

y_true = data["HeartDisease"]

min_max = data.agg(['min', 'max'])
print(min_max)
# train, test = train_test_split(data,test_size=0.2,random_state= 4)
# train_data_num = train[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]]

# fig = px.bar(data, x='ChestPainType', color='HeartDisease', title='Nombre de cas de maladie cardiaque par type de douleur thoracique')
# for column in data.columns[:-1]:
#     fig = px.histogram(data, x=column, color='HeartDisease', title=f'Distribution de {column} par maladie cardiaque')    
    
#     # Sauvegarder chaque graphique en tant qu'image PNG
#     fig.write_image(f'static/images/graphique_{column}_par_maladie_cardiaque.png')

app = Flask(__name__)

@app.route("/")
def index():
    dataHtlm = data.to_html(classes='table table-striped', header=True, max_rows=5).replace("<thead>", "<thead class='thead-dark'>", 1)
    return render_template('/index.html', vanillaData=dataHtlm)

@app.route("/logisticregression")
def logisticregression():
    return render_template('logisticReg.html')

@app.route("/randomforest")
def randomforest():
    return render_template('randomforest.html')

@app.route("/simulateur")
def simulateur():
    return render_template('simulateur.html')

@app.route("/simulateur/radomdata", methods=['POST', 'GET'])
def randomData():
    if request.method == 'POST':
        data = {
        'Age': np.random.randint(40, 80, 5),
        'Sex': np.random.choice([0, 1], size=5),
        'ChestPainType': np.random.choice([0, 1, 2, 3], size=5),
        'RestingBP': np.random.randint(90, 200, 5),
        'Cholesterol': np.random.randint(120, 600, 5),
        'FastingBS': np.random.choice([0, 1], size=5),
        'RestingECG': np.random.choice([0, 1, 2], size=5),
        'MaxHR': np.random.randint(70, 210, 5),
        'ExerciseAngina': np.random.choice([0, 1], size=5),
        'Oldpeak': np.random.uniform(0.0, 7.0, 5),
        'ST_Slope': np.random.choice([0, 1, 2], size=5),
        'ca': np.random.choice([0, 1, 2, 3, 4], size=5),
        'thal': np.random.choice([0, 1, 2, 3], size=5)
        }

        # Création du DataFrame
        df_random = pd.DataFrame(data)
        print(df_random)
        # Load the saved model
        loaded_model = joblib.load('data/randomforest_model.joblib')

        # Use the trained model to make predictions on the new data
        predictions = loaded_model.predict(df_random)        

        # Calculer l'exactitude (accuracy)
        # print(y_true)
        # accuracy = accuracy_score(y_true, predictions)

        # Afficher l'exactitude en pourcentage
        # accuracy_percentage = accuracy * 100
        # print(accuracy_percentage)
        df_random.insert(len(df_random.columns), 'Predictions', predictions)
        # df_random['Predictions'] = predictions

        # 'predictions' now contains the predicted values (0 or 1) indicating the presence or absence of heart disease
        print("Predictions for heart disease:\n", predictions)
        print(df_random.to_json())
    return df_random.to_json()

@app.route("/simulateur/mypredict", methods=['POST', 'GET'])
def myPredict():
    if request.method == "POST":
        data = request.form        
        params = {
        'Age': {"0" : data["age"]},
        'Sex': {"0" : data["sexe"]},
        'ChestPainType': {"0" : data["chestpaintype"]},
        'RestingBP': {"0" : data["restingbp"]},
        'Cholesterol': {"0" : data["cholesterol"]},
        'FastingBS': {"0" : data["fastingbs"]},
        'RestingECG': {"0" : data["restingecg"]},
        'MaxHR': {"0" : data["maxhr"]},
        'ExerciseAngina': {"0" : data["exerciseangina"]},
        'Oldpeak': {"0" : data["oldpeak"]},
        'ST_Slope': {"0" : data["stslope"]},
        'ca': {"0" : data["ca"]},
        'thal': {"0" : data["thal"]}
        }

        checkdata = all(value !="" for value in data.values())

        if checkdata:        
            df_random = pd.DataFrame(params)
            loaded_model = joblib.load('data/randomforest_model.joblib')
            
            predictions = loaded_model.predict(df_random)
            df_random.insert(len(df_random.columns), 'Predictions', predictions)        

            msg_predict = "aucun"
            if predictions[0] == 1:
                msg_predict = "forte chance"
            elif predictions[0] != 1 and predictions[0] != 0:
                msg_predict = "veuillez compléter tous les champs"

            print("Predictions for heart disease:\n", predictions, msg_predict)
        else:
            msg_predict = "veuillez compléter tous les champs"

    return jsonify(msg_predict)

if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)