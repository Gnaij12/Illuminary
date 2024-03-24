import sys 
sys.path.append("./Sentiment_Analysis")

from api_backend import create_driver, sign_into_facebook, webscrape_facebook_friends, webscrape_friend_posts
from training2 import process_string
import joblib

url = "https://www.facebook.com/profile.php?id=100005539517476"
model = joblib.load("./Sentiment_Analysis/model/model.pkl")
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