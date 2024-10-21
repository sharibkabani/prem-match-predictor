import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'premier-league-matches.csv'
data = pd.read_csv(file_path)
LINE = "-----------------------------------"

def preprocess_data(data):
    data['Result'] = data['FTR'].map({'H': 1, 'D': 0, 'A': -1})
    features = ['Home', 'Away', 'HomeGoals', 'AwayGoals']
    X = data[features]
    y = data['Result']
    le = LabelEncoder()
    X.loc[:, 'Home'] = le.fit_transform(X['Home'])
    X.loc[:, 'Away'] = le.transform(X['Away'])
    return X, y, le

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_prediction(model, X_test):
    return model.predict(X_test)

def head_to_head_stats(data, team1, team2):
    home_matches = data[(data['Home'] == team1) & (data['Away'] == team2)]
    away_matches = data[(data['Home'] == team2) & (data['Away'] == team1)]
    stats = {
        "team1": team1,
        "team2": team2,
        "team1_home_wins": (home_matches['Result'] == 1).sum(),
        "team2_away_wins": (home_matches['Result'] == -1).sum(),
        "draws": (home_matches['Result'] == 0).sum(),
        "team1_away_wins": (away_matches['Result'] == -1).sum(),
        "team2_home_wins": (away_matches['Result'] == 1).sum(),
        "team1_home_goals": home_matches['HomeGoals'].sum(),
        "team2_away_goals": home_matches['AwayGoals'].sum(),
        "team2_home_goals": away_matches['HomeGoals'].sum(),
        "team1_away_goals": away_matches['AwayGoals'].sum()
    }
    return stats

def predict_match_result(model, le, data, home_team, away_team):
    home_team_encoded = le.transform([home_team])[0]
    away_team_encoded = le.transform([away_team])[0]
    home_team_avg_goals = data[data['Home'] == home_team]['HomeGoals'].mean()
    away_team_avg_goals = data[data['Away'] == away_team]['AwayGoals'].mean()
    if pd.isna(home_team_avg_goals):
        home_team_avg_goals = 0
    if pd.isna(away_team_avg_goals):
        away_team_avg_goals = 0
    match_features = pd.DataFrame([[home_team_encoded, away_team_encoded, home_team_avg_goals, away_team_avg_goals]],
                                  columns=['Home', 'Away', 'HomeGoals', 'AwayGoals'])
    prediction = model.predict(match_features)
    stats = head_to_head_stats(data, home_team, away_team)
    result_map = {1: f"{home_team} is likely to win", 0: "It's likely to be a draw", -1: f"{away_team} is likely to win"}
    total_games = stats['team1_home_wins'] + stats['team2_away_wins'] + stats['draws'] + stats['team2_home_wins'] + stats['team1_away_wins']
    print("\nHead-to-Head Breakdown:")
    print(f"Total Games Played: {total_games}")
    print(LINE)
    print(f"{home_team} Home Wins: {stats['team1_home_wins']}")
    print(f"{away_team} Away Wins: {stats['team2_away_wins']}")
    print(f"Draws: {stats['draws']}")
    print(LINE)
    print(f"{away_team} Home Wins: {stats['team2_home_wins']}")
    print(f"{home_team} Away Wins: {stats['team1_away_wins']}")
    print(LINE)
    print(f"{home_team} Home Goals Scored: {stats['team1_home_goals']}")
    print(f"{away_team} Away Goals Scored: {stats['team2_away_goals']}")
    print(f"{away_team} Home Goals Scored: {stats['team2_home_goals']}")
    print(f"{home_team} Away Goals Scored: {stats['team1_away_goals']}")
    print(LINE)
    return result_map[prediction[0]]

def main():
    X, y, le = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred = make_prediction(model, X_test)
    print(LINE * 5)
    all_teams = le.classes_
    print("All Teams:")
    teams_per_row = 5
    for i in range(0, len(all_teams), teams_per_row):
        print(" | ".join(all_teams[i:i + teams_per_row]))
    print(LINE * 5)
    team1 = input("Enter the team 1 team: ")
    team2 = input("Enter the team 2 team: ")
    print(LINE * 5)
    print(f"Predicting the match result between Home: {team1} and Away: {team2}")
    prediction = predict_match_result(model, le, data, team1, team2)
    print(f"\nPrediction: {prediction}")
    print(LINE * 5)
    print(f"Predicting the match result between Home: {team2} and Away: {team1}")
    prediction = predict_match_result(model, le, data, team2, team1)
    print(f"\nPrediction: {prediction}")

if __name__ == "__main__":
    main()
