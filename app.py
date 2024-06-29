from flask import Flask, render_template, request, session, redirect, url_for,jsonify,flash
import pandas as pd  
import joblib
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import bcrypt
from flask_pymongo import PyMongo
from functools import wraps
from flask_mail import Mail, Message
import os
from bson import ObjectId
from flask_bcrypt import Bcrypt

app = Flask(__name__)

app.secret_key = b'\x9b\x1c\xe0\x9d\x1d\xaf\xc4\xec\xb4\xe8\xaf\xed\xcf\xb3\xea\xa4'  # Use a random secret key

app.config['MONGO_DBNAME'] = 'Pfe'
app.config['MONGO_URI'] = 'mongodb+srv://admin:QORuUa6PDqUfnIgj@cluster0.xktf2oc.mongodb.net/Pfe?retryWrites=true&w=majority'

mongo = PyMongo(app)


# Flask-Mail configuration
app.config['SECRET_KEY'] = "tsfyguaistyatuis589566875623568956"

app.config['MAIL_SERVER'] = "smtp.googlemail.com"

app.config['MAIL_PORT'] = 587

app.config['MAIL_USE_TLS'] = True

app.config['MAIL_USERNAME'] = "yassminezariat1@gmail.com"

app.config['MAIL_PASSWORD'] = "tpop mvct gtfq kpfa"

app.config['MAIL_DEFAULT_SENDER'] = "yassminezariat1@gmail.com" 

mail = Mail(app)


bcrypt = Bcrypt(app)

@app.route('/auth-forgot-password-basic', methods=['POST', 'GET'])
def forgotpassword():
    if request.method == 'POST':
        email = request.form['email']
        users = mongo.db.users
        user = users.find_one({'email': email})
        
        if user:
            reset_link = url_for('reset_password', email=email, _external=True)
            msg = Message('Password Reset Request', recipients=[email])
            msg.body = f'Please use the following link to reset your password: {reset_link}'
            mail.send(msg)
            flash('Un lien de réinitialisation du mot de passe a été envoyé à votre adresse e-mail.', 'info')
        else:
            flash('Adresse e-mail introuvable.', 'danger')
    return render_template('auth-forgot-password-basic.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    email = request.args.get('email')
    if not email:
        flash('Lien invalide ou expiré', 'danger')
        return redirect(url_for('forgotpassword'))
    
    users = mongo.db.users
    user = users.find_one({'email': email})
    
    if not user:
        flash('Lien invalide ou expiré', 'danger')
        return redirect(url_for('forgotpassword'))
    
    if request.method == 'POST':
        password1 = request.form['new-password']
        password2 = request.form['confirm-password']
        
        if password1 == password2:
            hashpass = bcrypt.generate_password_hash(password1).decode('utf-8')
            users.update_one({'email': email}, {'$set': {'password': hashpass}})
            flash('Votre mot de passe a été réinitialisé avec succès.', 'success')
            return redirect(url_for('authloginbasic'))
        else:
            flash('Les mots de passe ne correspondent pas', 'danger')
    
    return render_template('reset-password.html', email=email)

@app.route('/', methods=['POST', 'GET'])
@app.route('/auth-login-basic', methods=['POST', 'GET'])
def authloginbasic():
    if request.method == 'POST':
        users = mongo.db.users
        login_user = users.find_one({'username': request.form['email-username']}) or users.find_one({'email': request.form['email-username']})

        if login_user and bcrypt.check_password_hash(login_user['password'], request.form['password']):
            session['username'] = login_user['username']
            flash('Vous vous êtes connecté avec succès', 'success')
            return render_template('index.html')  # Assuming you have an 'index' route
        else:
            flash('Nom utilisateur/e-mail ou mot de passe invalide', 'danger')

    return render_template('auth-login-basic.html')


@app.route('/logout')
def logout():
    # Remove the username from the session if it's there
    session.pop('username', None)
    # Redirect to the login page or any other page after logout
    return redirect(url_for('authloginbasic'))



@app.route('/auth-register-basic', methods=['POST', 'GET'])
def authregisterbasic():
    if request.method == 'POST':
        users = mongo.db.users  # Ensure this line doesn't raise an error
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            if request.form['password1'] == request.form['password2']:
                hashpass = bcrypt.generate_password_hash(request.form['password1']).decode('utf-8')
                users.insert_one({
                    'username': request.form['username'],
                    'email': request.form['email'],
                    'password': hashpass
                })
                session['username'] = request.form['username']
                flash('Vous vous êtes inscrit avec succès', 'success')
                return redirect(url_for('authloginbasic'))
            else:
                flash('Les mots de passe ne correspondent pas', 'danger')
        else:
            flash('Ce nom d’utilisateur existe déjà', 'danger')
    return render_template('auth-register-basic.html')


@app.route('/index')
def first():
    return render_template('index.html')

@app.route('/index.html')
def retour():
    return render_template('index.html')

@app.route('/topics-detail.html')
def second():
    return render_template('topics-detail.html')

@app.route('/topics-listing.html')
def third():
    return render_template('topics-listing.html')

@app.route('/contact.html')
def fourth():
    return render_template('contact.html')

@app.route('/topics-detailGlycémie')
def dia():
    return render_template('topics-detailGlycémie.html')

@app.route('/topics-detailtrouAli')
def troublealimentaire():
    return render_template('topics-detailtrouAli.html')

@app.route('/problemesante')
def problemesante():
    return render_template('problemesante.html')

@app.route('/gestionsopk')
def gestionsopk():
    return render_template('gestionsopk.html')

@app.route('/styledevie')
def styledevie():
    return render_template('styledevie.html')

@app.route('/traitement')
def traitement():
    return render_template('traitement.html')

@app.route('/cds')
def cds():
    return render_template('cds.html')

@app.route('/regimediabete')
def regimediabete():
    return render_template('regimediabete.html')


@app.route('/endometrios')
def endometrios():
    return render_template('endometrios.html')


@app.route('/choix')
def choix_compli():
    return render_template('choix_compli.html')


@app.route('/complications')
def compli():
    return render_template('complications.html')

@app.route('/regimesopk')
def regimesopk():
    return render_template('regimesopk.html')

questions = [
    "Quel est votre âge en années ?",
    "Quel est votre poids en kilogrammes ?",
    "Quelle est votre taille en centimètres ?",
    "Votre indice de masse corporelle",
    "Quelle est votre fréquence cardiaque ?",
    "Quelle est votre fréquence respiratoire nombre de respirations par minute ?",
    "Avez-vous eu des irrégularités dans vos cycles menstruels? Si oui, veuillez préciser.",
    "Combien de jours dure habituellement votre cycle menstruel ?",
    "Depuis combien de temps êtes-vous mariée ?",
    "Êtes-vous enceinte en ce moment (oui/non) ?",
    "Combien de fois vous avez avorté",
    "Quelle est la taille de votre hanche en pouces ?",
    "Quelle est la taille de votre taille en pouces ?",
    "Votre rapport taille/hanche ?",
    "Avez-vous pris du poids récemment (oui/non) ?",
    "Avez-vous remarqué une croissance anormale des poils sur votre corps (oui/non) ?",
    "Avez-vous remarqué un assombrissement de votre peau (oui/non) ?",
    "Avez-vous remarqué une perte de cheveux (oui/non) ?",
    "Avez-vous des problèmes de peau tels que des boutons (oui/non) ?",
    "Consommez-vous régulièrement de la restauration rapide (oui/non) ?",
    "Faites-vous régulièrement de l'exercice (oui/non) ?",
    "Quelle est votre tension artérielle systolique (en mmHg) ?",
    "Quelle est votre tension artérielle diastolique (en mmHg) ?"
]

reponses_dataframe = pd.DataFrame(columns=['Question', 'Reponse'])


@app.route('/chat')
def index():
    session['question_index'] = -1  # Initialiser à -1 pour que la première question soit la question 0
    return render_template('chat.html')  


def preanalyse_pcos(reponses):
    pcos_suspecte = False
    
    # Convertir les réponses numériques nécessaires en nombres entiers ou flottants
    reponses_numeriques = []
    for x in reponses:
        if x.isdigit():
            reponses_numeriques.append(float(x))
        elif x.lower() == 'oui':
            reponses_numeriques.append(1)
        elif x.lower() == 'non':
            reponses_numeriques.append(0)
        else:
            try:
                reponses_numeriques.append(float(x))  # Tenter de convertir la valeur en nombre
            except ValueError:
                print(f"La valeur '{x}' ne peut pas être convertie en nombre.")
                reponses_numeriques.append(x)  # Conserver la valeur telle quelle
    
    # Votre logique de détection de PCOS avec les réponses numériques
    if isinstance(reponses_numeriques[0], float) and reponses_numeriques[0] >= 18 and isinstance(reponses_numeriques[5], float) and reponses_numeriques[5] >= 12:  # Âge adulte et fréquence respiratoire normale
        if reponses_numeriques[3] >= 25:  # IMC élevé (surpoids ou obésité)
            pcos_suspecte = True
        elif reponses_numeriques[6] > 2:  # Irrégularités menstruelles
            pcos_suspecte = True
        elif reponses_numeriques[8] >= 1 and reponses_numeriques[11] >= 1:  # Mariée depuis au moins 1 an et Avortement(s) signalé(s)
            pcos_suspecte = True
        elif 1 in reponses_numeriques[14:19]:  # Symptômes physiques
            pcos_suspecte = True
        elif reponses_numeriques[20] == 1:  # Consommation régulière de restauration rapide
            pcos_suspecte = True
        elif reponses_numeriques[22] >= 140 or reponses_numeriques[23] >= 90:  # Tension artérielle élevée
            pcos_suspecte = True

    if pcos_suspecte:
        result_message = "Il est conseillé de compléter l'évaluation approfondie de la PCOS 🩺."
    else:
        result_message = "Aucun signe précoce de PCOS n'a été détecté mais il est recommandé de compléter tout le test 🤔 ."
    
    return result_message



@app.route('/get', methods=['GET','POST'])
def chat():
    global reponses_dataframe

    if request.method == 'POST':
        msg = request.form['msg']

        if session['question_index'] == -1:
            welcome_message = "Bienvenue! Nous allons commencer notre premier diagnostic. Après avoir répondu aux questions, nous vous donnerons la probabilité d'avoir le SOPK. Le questionnaire commence maintenant, veuillez nous donner des réponses correctes pour un diagnostic précis."
            session['question_index'] += 1
            return welcome_message + "\n\n" + questions[0]

        question_actuelle = questions[session['question_index']]

        if '(oui/non)' in question_actuelle:
            if msg.lower() not in ['oui', 'non']:
                return "Pour la question qui contient '(oui/non)', veuillez répondre par 'oui' ou 'non'."
            else:
                session['question_index'] += 1
                if session['question_index'] < len(questions):
                    new_row = pd.DataFrame({'Question': [question_actuelle], 'Reponse': [msg.lower()]})
                    reponses_dataframe = pd.concat([reponses_dataframe, new_row], ignore_index=True)
                    return questions[session['question_index']]
                else:
                    new_row = pd.DataFrame({'Question': [question_actuelle], 'Reponse': [msg.lower()]})
                    reponses_dataframe = pd.concat([reponses_dataframe, new_row], ignore_index=True)
                    result_message = preanalyse_pcos(reponses_dataframe['Reponse'].tolist())  
                    return ""  

        try:
            reponse_numerique = float(msg.lower())
        except ValueError:
            return "Veuillez entrer une réponse numérique valide."

        if session['question_index'] < len(questions):
            new_row = pd.DataFrame({'Question': [question_actuelle], 'Reponse': [msg.lower()]})
            reponses_dataframe = pd.concat([reponses_dataframe, new_row], ignore_index=True)
            session['question_index'] += 1
            if session['question_index'] < len(questions):
                return questions[session['question_index']]
            else:
                result_message = preanalyse_pcos(reponses_dataframe['Reponse'].tolist())  
                return ""  

@app.route('/resultat')
def afficher_resultat():
    result_message = preanalyse_pcos(reponses_dataframe['Reponse'].tolist())  
    if result_message == "Il est conseillé de compléter l'évaluation approfondie de la PCOS 🩺.":
        #return redirect(url_for('autre_page'))
        return render_template('resultat_evaluation.html', reponses_dataframe=reponses_dataframe.to_html(), result_message=result_message)

    else:
        return render_template('resultat_evaluation.html', reponses_dataframe=reponses_dataframe.to_html(), result_message=result_message)

@app.route('/analysepcos')
def autre_page():
    # Traitement à effectuer sur l'autre page
    return render_template('analysepcos.html')

#model = joblib.load('XGBClassifier.pkl')
model = joblib.load('gb.pkl')
@app.route('/analyser', methods=['POST'])
def analyser():
    global reponses_dataframe
    # Récupérer les données du formulaire
    form_data = request.form.to_dict()
    # Ajouter les nouvelles données au dataframe existant
    new_row = pd.DataFrame.from_dict(form_data, orient='index', columns=['Reponse']).reset_index()
    new_row.columns = ['Question', 'Reponse']  
    reponses_dataframe = pd.concat([reponses_dataframe, new_row], ignore_index=True)
    
    # Liste des questions dans l'ordre souhaité
    ordre_questions = [
    "Quel est votre âge en années ?", 
    "Quel est votre poids en kilogrammes ?", 
    "Quelle est votre taille en centimètres ?", 
    "Votre indice de masse corporelle", 
    "Groupe sanguin", 
    "Quelle est votre fréquence cardiaque ?",
    "Quelle est votre fréquence respiratoire nombre de respirations par minute ?", 
    "Hb(g/dl)", 
    "Avez-vous eu des irrégularités dans vos cycles menstruels? Si oui, veuillez préciser.", 
    "Combien de jours dure habituellement votre cycle menstruel ?", 
    "Depuis combien de temps êtes-vous mariée ?", 
    "Êtes-vous enceinte en ce moment (oui/non) ?", 
    "Combien de fois vous avez avorté", 
    "beta_hcg_I", 
    "beta_hcg_II", 
    "fsh", 
    "lh", 
    "fsh_lh", 
    "Quelle est la taille de votre hanche en pouces ?", 
    "Quelle est la taille de votre taille en pouces ?", 
    "Votre rapport taille/hanche ?", 
    "tsh", 
    "amh", 
    "prl", 
    "vit_d3", 
    "prg", 
    "rbs", 
    "Avez-vous pris du poids récemment (oui/non) ?", 
    "Avez-vous remarqué une croissance anormale des poils sur votre corps (oui/non) ?", 
    "Avez-vous remarqué un assombrissement de votre peau (oui/non) ?", 
    "Avez-vous remarqué une perte de cheveux (oui/non) ?", 
    "Avez-vous des problèmes de peau tels que des boutons (oui/non) ?", 
    "Consommez-vous régulièrement de la restauration rapide (oui/non) ?", 
    "Faites-vous régulièrement de l'exercice (oui/non) ?", 
    "Quelle est votre tension artérielle systolique (en mmHg) ?", 
    "Quelle est votre tension artérielle diastolique (en mmHg) ?", 
    "follicle_l", 
    "follicle_r", 
    "avg_f_size_l", 
    "avg_f_size_r", 
    "endometrium"
    ]
    
    # Créer une liste ordonnée de réponses correspondant à l'ordre des questions
    reponses_ordre = []
    for question in ordre_questions:
        reponse = reponses_dataframe.loc[reponses_dataframe['Question'] == question, 'Reponse'].values[0]
        reponses_ordre.append(reponse)
    
    
    reponses_ordre = [1 if reponse.lower() == 'oui' else 0 if reponse.lower() == 'non' else float(reponse) for reponse in reponses_ordre]
    print(reponses_ordre)
    # Convertir les réponses ordonnées en un tableau NumPy pour la prédiction
    features = np.array(reponses_ordre).reshape(1, -1)
    
    # Effectuer la prédiction avec les données combinées
    prediction = model.predict(features)
    
    # Gérer le résultat de la prédiction
    if prediction == 0:
        print("Pas de PCOS")
        return render_template('resultatsang.html')
    else:
        print("PCOS")
        return render_template('prediction_eco.html')


@app.route('/analyser_diabete')
def Diabetes():
    # Traitement à effectuer sur l'autre page
    return render_template('Diabetes.html')

@app.route('/analyser_diabete', methods=['POST'])
def analyser_diabete():

    global reponses_dataframe

    model_Diabetes = load_model('Diabetes_Prediction.h5')
    #model_Diabetes = pickle.load(open('Diabetes_Prediction.pkl', 'rb'))

    # Ordre des questions pour l'analyse du diabète
    ordre_questions_diabete = [
        "Combien de fois avez-vous été enceinte ?", 
        "Glucose", 
        "Pression_artérielle",  
        "Épaisseur_de_la_peau",  
        "Insuline", 
        "Votre indice de masse corporelle", 
        "Fonction_de_pedigree_du_diabète", 
        "Quel est votre âge en années ?"
    ]

    # Récupérer les données du formulaire
    form_data = request.form.to_dict()

    # Ajouter les nouvelles données au DataFrame existant
    new_row = pd.DataFrame.from_dict(form_data, orient='index', columns=['Reponse']).reset_index()
    new_row.columns = ['Question', 'Reponse']  
    reponses_dataframe = pd.concat([reponses_dataframe, new_row], ignore_index=True)

    # Réorganiser les réponses dans l'ordre spécifié pour le diabète
    reponses_dataframe = reponses_dataframe[reponses_dataframe['Question'].isin(ordre_questions_diabete)]

    # Assurez-vous que les réponses sont dans le bon format (nombre ou oui/non)
    reponses_array_diabete = []

    for question in ordre_questions_diabete:
        reponse = reponses_dataframe[reponses_dataframe['Question'] == question]['Reponse'].values[0]
        if reponse.lower() == 'oui':
            reponses_array_diabete.append(1)
        elif reponse.lower() == 'non':
            reponses_array_diabete.append(0)
        else:
            reponses_array_diabete.append(float(reponse))
   # Afficher le contenu du DataFrame et son nombre d'entrées
    print(reponses_array_diabete)
    print("Nombre d'entrées dans le DataFrame : ", len(reponses_array_diabete))
    # Effectuer la prédiction avec les données combinées
    features_diabete = np.array(reponses_array_diabete).reshape(1, -1)
    prediction_diabete = model_Diabetes.predict(features_diabete)

    # Determine the result message based on the prediction
    if np.isin(0, np.argmax(prediction_diabete, axis=-1)):
        result_message_diabete= "Vous êtes normal."
    elif np.isin(1, np.argmax(prediction_diabete, axis=-1)):
        result_message_diabete= "Vous êtes prédiabétique."
    else:
        result_message_diabete = "Vous êtes diabétique."

    # Ajouter le message de résultat au contexte de rendu pour la page Diabetes.html
    return jsonify({'result_message_diabete': result_message_diabete})

# Charger le modèle Keras
new_model = load_model('model_PCO_detect.h5')

def test_model(model, img_path):
    img = image.load_img(img_path, color_mode="rgb", target_size=(225, 225))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=-1)  # Ajout de la dimension du canal pour correspondre aux attentes du modèle
    predictions = model.predict(x).argmax()
    # Dictionnaire de correspondance entre les prédictions et les émotions
    pco_labels = {0: 'SOPK détecté', 1: 'SOPK non détecté'}
    return pco_labels[predictions]

@app.route('/prediction_eco', methods=['GET'])
def prediction_eco():
    return render_template('prediction_eco.html')

@app.route('/prediction_eco', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            pred = test_model(new_model, file_path)
            return jsonify({'prediction': pred})  # Passer la prédiction au template
    return jsonify({'error': 'Aucun fichier n\'a été téléchargé.'})



# Define the features order
features_order = [
    'Painful Periods', 'Heavy / Extreme menstrual bleeding', 'Fatigue / Chronic fatigue', 
    'Pelvic pain', 'Infertility', 'Hormonal problems', 'Abdominal pain / pressure', 
    'Extreme / Severe pain', 'Painful bowel movements', 'Lower back pain', 'Cramping', 
    'Irregular / Missed periods', 'Acne / pimples', 'Painful / Burning pain during sex (Dyspareunia)', 
    'Decreased energy / Exhaustion', 'Pain / Chronic pain', 'Ovarian cysts', 'Headaches', 
    'Bowel pain', 'Loss of appetite', 'Bloating', 'Back pain', 'Diarrhea', 'Bleeding', 
    'Vomiting / constant vomiting', 'Fever', 'Excessive bleeding', 'Long menstruation', 
    'Migraines', 'Stomach cramping', 'Vaginal Pain/Pressure', 'Depression / Anxiety', 
    'Mood swings', 'Nausea', 'Menstrual clots', 'IBS-like symptoms', 'Abnormal uterine bleeding'
]

@app.route('/analyser_endometriose', methods=['POST'])
def analyser_endometriose():
    X_train = np.random.rand(100, len(features_order))
    model = joblib.load('random_forest_model.pkl')

# Ajuster le scaler sur les données d'entraînement
    scaler = StandardScaler()
    scaler.fit(X_train)
    # Map form data to the feature order
    form_data = request.form.to_dict()

    # Ensure all features are present, setting missing features to 0
    feature_values = []
    for feature in features_order:
        feature_key = feature.replace(' / ', '_').replace(' ', '_').replace('(', '').replace(')', '').lower()
        feature_values.append(1 if form_data.get(feature_key) == 'on' else 0)
    
    # Convert to DataFrame for consistency
    input_data = pd.DataFrame([feature_values], columns=features_order)
    print(input_data)

    # Scale the data
    input_data_scaled = scaler.transform(input_data)
    # Make the prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)[:, 1]
    
    result_message_endometriose = "Les résultats indiquent que vous pourriez être atteinte d'endométriose." if prediction[0] == 1 else "Les résultats indiquent que vous ne présentez pas de signes d'endométriose."
    return jsonify({'result_message_endometriose': result_message_endometriose, 'result_probability': prediction_proba[0]})


if __name__ == '__main__':
    app.run(debug=True)
