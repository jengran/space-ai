import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the asteroid dataset
print("Loading NASA asteroid data...")
df = pd.read_csv("nasa.csv")
print(f"Total asteroids loaded: {len(df)}")

# Pick the features our AI will learn from
features = [
    'Est Dia in KM(min)',
    'Est Dia in KM(max)',
    'Relative Velocity km per sec',
    'Miss Dist.(kilometers)',
    'Absolute Magnitude'
]

# X = the measurements, y = hazardous or not
X = df[features]
y = df['Hazardous']

# Split data — 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining asteroids: {len(X_train)}")
print(f"Testing asteroids: {len(X_test)}")

# Train the AI model
print("\nTraining AI model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("AI model trained!!")

# Test the AI model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAI Accuracy: {accuracy * 100:.2f}%")

# Show feature importance chart
print("\nGenerating chart...")
importance = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importance, color='#4CC9F0')
plt.xlabel('Importance')
plt.title('What factors does the AI use to detect hazardous asteroids?')
plt.tight_layout()
plt.savefig('asteroid_ai_chart.png')
plt.show()
print("Chart saved as asteroid_ai_chart.png!")

# Test with a made up asteroid
print("\n--- Testing with a new asteroid ---")
new_asteroid = pd.DataFrame([{
    'Est Dia in KM(min)': 0.5,
    'Est Dia in KM(max)': 1.2,
    'Relative Velocity km per sec': 25.0,
    'Miss Dist.(kilometers)': 500000,
    'Absolute Magnitude': 18.5
}])

result = model.predict(new_asteroid)
confidence = model.predict_proba(new_asteroid)[0]

print(f"Asteroid size: 0.5 - 1.2 km")
print(f"Speed: 25 km/sec")
print(f"Distance from Earth: 500,000 km")
print(f"AI Prediction: {'HAZARDOUS' if result[0] else 'NOT HAZARDOUS'}")
print(f"Confidence: {max(confidence) * 100:.1f}%")