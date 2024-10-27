from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import sqlite3

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Database function to retrieve player data based on player IDs
def get_player_features(player_ids):
    conn = sqlite3.connect('valorant.sqlite')
    cursor = conn.cursor()
    features = []

    for player_id in player_ids:
        cursor.execute("SELECT * FROM Game_Scoreboard WHERE player_id = ?", (player_id,))
        player_data = cursor.fetchone()
        
        if player_data:
            # Assuming player data starts from index 1 (skip player_id column)
            features.extend(player_data[1:])
        else:
            conn.close()
            raise ValueError(f"No data found for player ID {player_id}")

    conn.close()
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get player IDs for both teams from the form data
        team1_ids = request.form.getlist('team1_ids[]')
        team2_ids = request.form.getlist('team2_ids[]')
        print("Team 1 IDs:", team1_ids)
        print("Team 2 IDs:", team2_ids)
        # Validate that exactly 5 IDs are provided for each team
        if len(team1_ids) != 5 or len(team2_ids) != 5:
            return jsonify({'error': 'Each team must have exactly 5 player IDs.'}), 400
        # Retrieve features for both teams from the database
        team1_features = get_player_features(team1_ids)
        team2_features = get_player_features(team2_ids)

        # Concatenate both team features into a single array for model prediction
        match_data = np.hstack([team1_features, team2_features])

        # Ensure match_data is a float32 NumPy array and reshape for model input
        match_data = np.array(match_data, dtype=np.float32).reshape(1, -1)

        # Check for NaN or infinite values in match_data
        if np.isnan(match_data).any() or np.isinf(match_data).any():
            return jsonify({'error': 'Invalid data found in player features. Please check input values.'})

        # Predict the winner
        prediction = model.predict(match_data)
        winner = "Team 1" if prediction[0][0] > 0.5 else "Team 2"

        return jsonify({'winner': winner})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({'error': 'An error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
