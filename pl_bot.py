import requests
import pandas as pd
import pulp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def fetch_data():
    # Fetch data from the FPL API
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        players = pd.DataFrame(data['elements'])
        #print(players.columns) !!! debugging
        players['name'] = players['first_name'] + ' ' + players['second_name']
        required_columns = ['name', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence', 'creativity',
                            'threat', 'ict_index', 'total_points', 'now_cost', 'team', 'element_type']
        players = players[required_columns]
        return players
    else:
        print("Failed to fetch data from FPL API")
        return None

def preprocess_and_validate_data(players_df):
    # Print data types before preprocessing
    print("Data Types before preprocessing:")
    print(players_df.dtypes)

    # Convert convertible columns to numeric types
    numeric_features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence',
                        'creativity', 'threat', 'ict_index', 'now_cost', 'team_encoded', 'element_type_encoded']

    for feature in numeric_features:
        if players_df[feature].dtype == 'object':
            # Attempt to convert to numeric, coerce invalid values to NaN
            players_df[feature] = pd.to_numeric(players_df[feature], errors='coerce')

    # Checking for missing or invalid (NaN) values and handling them
    if players_df[numeric_features].isnull().any().any():
        print("Missing values detected after conversion. Filling missing values with default (e.g., 0).")
        players_df[numeric_features] = players_df[numeric_features].fillna(0)

    print("Data inspection complete. No missing values should be present now.")
    return players_df


def train_model(players):
    # Prepare data for machine learning
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence', 'creativity', 'threat', 'ict_index', 'now_cost']
    target = 'total_points'

    #exclude position (element_type) and team from training because they are strings
    #we do this by encoding these categorical variables
    if 'team' in players.columns:
        label_encoder_team = LabelEncoder()
        players['team_encoded'] = label_encoder_team.fit_transform(players['team'])
        features.append('team_encoded')

    if 'element_type' in players.columns:
        label_encoder_position = LabelEncoder()
        players['element_type_encoded'] = label_encoder_position.fit_transform(players['element_type'])
        features.append('element_type_encoded')

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

    return model, features


def predict_for_player(model, player_row, features):
    # Ensure player_row is in DataFrame form with the correct feature names
    player_stats_df = pd.DataFrame([player_row[features]])

    # Make prediction
    predicted_points = model.predict(player_stats_df)[0]
    return predicted_points


def search_player_screen(model, players, features):
    while True:
        # Get the player's name from the user
        search_name = input(
            "Enter a player's name to predict their fantasy points (or type 'back' to return to the menu): ")

        if search_name.lower() == 'back':
            break

        # Search for players with matching names
        matching_players = players[players['name'].str.contains(search_name, case=False)]

        if matching_players.empty:
            print(f"No players found matching '{search_name}'. Please try again.")
        else:
            # Display matching players and ask user to select
            print("Matching players:")
            for idx, player in enumerate(matching_players['name'], 1):
                print(f"{idx}. {player}")

            try:
                selection = int(input("Select a player by number (or type '0' to cancel): "))
                if selection == 0:
                    print("Cancelled selection.")
                    continue

                selected_player_row = matching_players.iloc[selection - 1]
                selected_player_name = selected_player_row['name']

                # Predict fantasy points for the selected player
                predicted_points = predict_for_player(model, selected_player_row, features)
                print(f"Predicted Fantasy Points for {selected_player_name}: {predicted_points:.2f}")

            except (ValueError, IndexError):
                print("Invalid selection. Please try again.")


#Predict the best team for next gameweek with pulp
def recommend_best_team(players_df, model, budget=100):
    # Preprocess and validate data
    players_df = preprocess_and_validate_data(players_df)

    # Prepare features for prediction
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence', 'creativity', 'threat', 'ict_index',
                'now_cost', 'team_encoded', 'element_type_encoded']
    X = players_df[features]

    # Predict fantasy points for each player
    predictions = model.predict(X)
    players_df['predicted_points'] = predictions

    # Initialize the optimization problem
    problem = pulp.LpProblem("Fantasy_Team_Selection", pulp.LpMaximize)

    # Create binary variables for each player, and explictly set them to be binary
    player_vars = {i: pulp.LpVariable(f"player_{i}", cat="Binary") for i in players_df.index}

    # Objective function: Maximize predicted points
    problem += pulp.lpSum(
        [player_vars[i] * players_df.loc[i, 'predicted_points'] for i in players_df.index]), "Total Predicted Points"

    # Constraints
    problem += pulp.lpSum(
        [player_vars[i] * players_df.loc[i, "now_cost"] for i in players_df.index]) <= budget, "Total Cost Constraint"

    constraints = [
        # Team Size Constraint
        (pulp.lpSum([player_vars[i] for i in players_df.index]) == 15, "Team Size Constraint"),
        # Constraints by Position
        (pulp.lpSum([player_vars[i] for i in players_df[players_df['element_type'] == 'GK'].index]) == 2,
         "GK Constraint"),
        (pulp.lpSum([player_vars[i] for i in players_df[players_df['element_type'] == 'DEF'].index]) >= 3,
         "Min DEF Constraint"),
        (pulp.lpSum([player_vars[i] for i in players_df[players_df['element_type'] == 'DEF'].index]) <= 5,
         "Max DEF Constraint"),
        (pulp.lpSum([player_vars[i] for i in players_df[players_df['element_type'] == 'MID'].index]) >= 3,
         "Min MID Constraint"),
        (pulp.lpSum([player_vars[i] for i in players_df[players_df['element_type'] == 'MID'].index]) <= 5,
         "Max MID Constraint"),
        (pulp.lpSum([player_vars[i] for i in players_df[players_df['element_type'] == 'FWD'].index]) >= 1,
         "Min FWD Constraint"),
        (pulp.lpSum([player_vars[i] for i in players_df[players_df['element_type'] == 'FWD'].index]) <= 3,
         "Max FWD Constraint"),
    ]

    for team in players_df['team_encoded'].unique():
        constraints.append((pulp.lpSum(
            [player_vars[i] for i in players_df[players_df['team_encoded'] == team].index]) <= 3,
                            f"Max Players per Team ({team})"))

    # Add constraints incrementally and debug
    for constraint, name in constraints:
        problem += constraint, name
        problem.solve()
        status = pulp.LpStatus[problem.status]
        if status != "Optimal":
            print(f"Problem became infeasible after adding constraint: {name}")
            print(f"Status: {status}")
            break

    if pulp.LpStatus[problem.status] == "Optimal":
        selected_players = [i for i in players_df.index if player_vars[i].varValue == 1]
        return players_df.loc[selected_players]
    else:
        print("No optimal solution found. Detailed debug information follows:")
        for name, constraint in problem.constraints.items():
            print(f"{name}: {constraint.value()}")

        player_costs_selections = {i: (players_df.loc[i, 'now_cost'], player_vars[i].varValue) for i in
                                   players_df.index}

        print("Player costs and selection status:")
        for idx, (cost, selected) in player_costs_selections.items():
            # Print both cost and selected status, ensuring selected values are checked
            if selected not in [0, 1]:
                print(f"Warning: Player {idx} has an invalid selection value: {selected}")
            print(f"Player {idx}: Cost = {cost}, Selected = {selected}")

        total_cost = sum(players_df.loc[i, 'now_cost'] * player_vars[i].varValue for i in players_df.index)
        print(f"Total Cost: {total_cost} (Budget: {budget})")
        print(f"Objective Value: {pulp.value(problem.objective)}")

        return None


def menu():
    print("Welcome to the Fantasy Premier League Prediction Bot!")
    print("1. Predict fantasy points for a specific player")
    print("2. Predict best team for next GameWeek (under development")
    print("2. Exit")
    choice = input("Please select an option (1, 2, 3): ")
    return choice


def main():
    # Fetch and train model data
    players = fetch_data()
    if players is None:
        return

    model, features = train_model(players)

    while True:
        #show the menu and get user's choice
        choice = menu()

        if choice == '1':
            # Go to search for a player screen
            search_player_screen(model, players, features)
        elif choice == '2':
            #Go to team recommendation screen
            recommend_best_team(players, model)
        elif choice == '3':
            # Exit the program
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
