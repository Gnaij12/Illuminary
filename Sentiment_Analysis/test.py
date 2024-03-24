import sys
sys.path.append("./Sentiment_Analysis")
from training3 import process_string
import joblib
import numpy as np
model = joblib.load("./Sentiment_Analysis/models/model.pkl")
s = process_string("It's just over 2 years since I was diagnosed with #anxiety and #depression")
print(s)
print(np.average(model.predict(s)).round(0))
print()
