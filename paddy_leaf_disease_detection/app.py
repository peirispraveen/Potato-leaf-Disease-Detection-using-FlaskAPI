from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from paddy_disease_detection import detect_rice_diseases

app = Flask(__name__)


@app.route('/')
def home():  # put application's code here
    return render_template("home.html")

@app.route('/disease_prediction', methods=['Post'])
def disease_prediction():
    if request.method == 'POST':
        f = request.files['paddy_image']
        full_name = secure_filename(f.filename)
        leaf_image = 'E:/AIDS Y2/prac/paddy_leaf_disease_detection/static/user_files/' + full_name
        f.save(leaf_image)
        prediction_plot = detect_rice_diseases(leaf_image)
    return render_template('paddy_prediction.html', user_image="/static/user_files/" + full_name
                           , predicted_image=prediction_plot)

if __name__ == '__main__':
    app.run()
