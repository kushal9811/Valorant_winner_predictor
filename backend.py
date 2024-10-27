from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import sqlite3

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Function to retrieve and aggregate player stats for a given team based on player IDs
def get_team_stats(player_ids):
    # Connect to the database to retrieve data
    conn = sqlite3.connect('valorant.sqlite')
    
    # Load the scoreboard data for specified player IDs
    query = f"SELECT * FROM Game_Scoreboard WHERE PlayerID IN ({','.join(['?']*len(player_ids))})"
    df_scoreboard = pd.read_sql_query(query, conn, params=player_ids)
    
    # Close the database connection
    conn.close()
    
    # Clean the data by removing rows with missing values
    df_scoreboard = df_scoreboard.dropna()
    
    # Ensure all relevant columns are of numeric type
    columns_to_use = ['ACS', 'Kills', 'Deaths', 'Assists', 'PlusMinus', 
                      'KAST_Percent', 'ADR', 'HS_Percent', 'FirstKills', 
                      'FirstDeaths', 'FKFD_PlusMinus', 'Num_2Ks', 
                      'Num_3Ks', 'Num_4Ks', 'Num_5Ks']
    df_scoreboard[columns_to_use] = df_scoreboard[columns_to_use].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with any remaining non-numeric values
    df_scoreboard = df_scoreboard.dropna()

    # Aggregate the statistics by averaging the player's stats
    team_stats_mean = df_scoreboard[columns_to_use].mean()
    
    return team_stats_mean

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

        # Retrieve and aggregate stats for both teams
        team1_stats = get_team_stats(team1_ids)
        team2_stats = get_team_stats(team2_ids)

        # Check if data retrieval resulted in empty stats (e.g., due to missing data)
        if team1_stats.isnull().all() or team2_stats.isnull().all():
            return jsonify({'error': 'Invalid data found for one or both teams. Please check the database.'})

        # Concatenate the statistics for both teams into a single feature vector
        match_features = pd.concat([team1_stats, team2_stats], axis=0).values.reshape(1, -1)

        # Check for NaN or infinite values in match_features
        if np.isnan(match_features).any() or np.isinf(match_features).any():
            return jsonify({'error': 'Invalid data found in player features. Please check input values.'})

        # Predict the winner using the trained DNN model
        prediction = model.predict(match_features)
        winner = "Team 1" if prediction[0][0] > 0.5 else "Team 2"

        return jsonify({'winner': winner})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({'error': 'An error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
