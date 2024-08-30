import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def fetch_data():
    # Fetch data from the FPL API
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        players = pd.DataFrame(data['elements'])
        players['name'] = players['first_name'] + ' ' + players['second_name']
        required_columns = ['name', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence', 'creativity',
                            'threat', 'ict_index', 'total_points']
        players = players[required_columns]
        return players
    else:
        print("Failed to fetch data from FPL API")
        return None


def train_model(players):
    # Prepare data for machine learning
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence', 'creativity', 'threat', 'ict_index']
    target = 'total_points'

    X = players[features]
    y = players[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

    return model, players


def predict_for_player(model, player_name, players):
    # Search for the player by name
    player_row = players[players['name'].str.contains(player_name, case=False)]

    if not player_row.empty:
        # Get player stats
        player_stats = player_row[
            ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence', 'creativity', 'threat', 'ict_index']]

        # Make prediction
        predicted_points = model.predict(player_stats)[0]

        return predicted_points
    else:
        return None


def main():
    players = fetch_data()
    if players is None:
        return

    model, players = train_model(players)

    while True:
        # Get the player's name from the user
        player_name = input("Enter a player's name to predict their fantasy points (or type 'exit' to quit): ")

        if player_name.lower() == 'exit':
            break

        # Predict fantasy points for the specified player
        predicted_points = predict_for_player(model, player_name, players)

        if predicted_points is not None:
            print(f"Predicted Fantasy Points for {player_name}: {predicted_points:.2f}")
        else:
            print(f"Player '{player_name}' not found in the dataset.")


if __name__ == "__main__":
    main()
