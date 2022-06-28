import pickle
from flask import render_template, request,Flask,jsonify
import numpy as np

app = Flask(__name__)
clf = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")  


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        ai = request.form.get("age")
        wi = request.form.get("weight")
        temp_array = [int(ai),int(wi)]
        data = np.array([temp_array])
        predic = int(clf.predict(data)[0])
        return render_template("index.html", msg = predic)  

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = clf.predict([np.array(list(data.values()))])
    # output = prediction[0]
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
