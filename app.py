from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    vect_news = vectorizer.transform([news])
    prediction = model.predict(vect_news)[0]
    return jsonify({"prediction": "Fake News" if prediction == 0 else "Real News"})

if __name__ == "__main__":
    app.run(debug=True)
