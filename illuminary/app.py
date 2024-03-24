from flask import Flask, render_template, request, redirect, jsonify
import re
import sys 
import time
sys.path.append("./Sentiment_Analysis")

from api_backend import create_driver, sign_into_facebook, webscrape_facebook_friends, webscrape_friend_posts
from training2 import process_string
import joblib
from numpy import average

app = Flask(__name__)

model = joblib.load("./Sentiment_Analysis/models/model.pkl")

@app.route('/')
def index():
    return render_template('page1.html')


@app.route('/single', methods=["GET"])
def single():
    url = request.args.get("url")
    friend = request.args.get("friend")
    time.sleep(5)
    friend_posts = webscrape_friend_posts(url, friend)
    # total_mental_health_score = 0
    data = {}
    for post in friend_posts:
        processed_post = process_string(post)
        data[post] = int(average(model.predict(processed_post)))
        # total_mental_health_score += average(model.predict(post)).round(0)

    return data


@app.route('/submit', methods=["GET", "POST"])
def character():
    url = request.form["url"]

    # url = "https://www.facebook.com/profile.php?id=100005539517476"
    friend_list = webscrape_facebook_friends(url)
    
    return render_template("page2.html",friends=friend_list, url=url)

@app.route("/thanks")
def thanks():
    return render_template("thanks.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)