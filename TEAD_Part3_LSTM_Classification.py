
# === PART 3: LSTM-Based Classification and Intrusion Decision ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Select relevant features (can be trust + other communication metrics)
features = ['CLR', 'CWR', 'CFD']
X = df[features].values
y = df['class'].values

# Reshape input to fit LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
