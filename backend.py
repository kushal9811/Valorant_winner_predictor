from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import sqlite3

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Function to retrieve and aggregate player stats for a given team based on player IDs
def get_team_stats(player_ids):
    conn = sqlite3.connect('valorant.sqlite')
    query = f"SELECT * FROM Game_Scoreboard WHERE PlayerID IN ({','.join(['?']*len(player_ids))})"
    df_scoreboard = pd.read_sql_query(query, conn, params=player_ids)
    conn.close()
    df_scoreboard = df_scoreboard.dropna()
    
    columns_to_use = ['ACS', 'Kills', 'Deaths', 'Assists', 'PlusMinus', 
                      'KAST_Percent', 'ADR', 'HS_Percent', 'FirstKills', 
                      'FirstDeaths', 'FKFD_PlusMinus', 'Num_2Ks', 
                      'Num_3Ks', 'Num_4Ks', 'Num_5Ks']
    df_scoreboard[columns_to_use] = df_scoreboard[columns_to_use].apply(pd.to_numeric, errors='coerce')
    df_scoreboard = df_scoreboard.dropna()
    
    team_stats_mean = df_scoreboard[columns_to_use].mean()
    return team_stats_mean

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        team1_ids = request.form.getlist('team1_ids[]')
        team2_ids = request.form.getlist('team2_ids[]')
        
        if len(team1_ids) != 5 or len(team2_ids) != 5:
            return render_template('index.html', error='Each team must have exactly 5 player IDs.')
        
        team1_stats = get_team_stats(team1_ids)
        team2_stats = get_team_stats(team2_ids)
        
        match_features = (team1_stats - team2_stats).values.reshape(1, -1).astype(np.float32)
        
        prediction = model.predict(match_features)
        winner = "Team 1" if prediction[0][0] > 0.5 else "Team 2"

        # Render the result in `result.html`
        return render_template('result.html', winner=winner)

    except Exception as e:
        return render_template('index.html', error='An error occurred. Please try again later.')

if __name__ == '__main__':
    app.run(debug=True)
