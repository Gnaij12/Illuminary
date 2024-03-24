import sqlite3
from flask import Flask, render_template
import sys 
sys.path.append("./illuminary/illuminary")
sys.path.append("./Sentiment_Analysis")

from api_backend import create_driver, sign_into_facebook, webscrape_facebook_friends, webscrape_friend_posts
from training2 import process_string
import joblib

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    return render_template('thing.html')
    # conn = get_db_connection()
    # data = conn.execute('SELECT * FROM characters').fetchall()
    # conn.close()
    
    # return render_template('characters.html', characters=data)

@app.route('/submit?<url>')
def character(url):
    url = "https://www.facebook.com/profile.php?id=100005539517476"
    model = joblib.load("./Sentiment_Analysis/model/model.pkl")
    driver = create_driver()
    sign_into_facebook(driver)
    friend_list = webscrape_facebook_friends(url)
    worry_list = []
    for friend in friend_list:
        friend_posts = webscrape_friend_posts(url, friend)
        total_mental_health_score = 0
        for post in friend_posts:
            post = process_string(post)
            total_mental_health_score += model.predict(post)
        if total_mental_health_score / len(friend_posts) > 5:
            worry_list.append((friend, total_mental_health_score / len(friend_posts)))
    return render_template("page2.html",stats=worry_list)

@app.route("/thanks")
def thanks():
    return render_template("thanks.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)