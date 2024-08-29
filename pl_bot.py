import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#fetch data from the FPL API
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
response = requests.get(url)

if response.status_code == 200:
    #create a DataFrame from the fetched data
    data = response.json()
    players = pd.DataFrame(data['elements'])

    #combine first and last names in our DataFrame for later use
    players['name'] = players['first_name'] + ' ' + players['second_name']

    #clean and select relevant columns
    players = players[['name', 'minutes', 'goals_scored', 'assists',
                       'clean_sheets', 'influence', 'creativity', 'threat', 'ict_index', 'total_points']]

    #prepare our data for machine learning
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'influence', 'creativity', 'threat', 'ict_index']
    #fantasy points to predict
    target = 'total_points'

    X = players[features]
    y = players[target]
    names = players['name']

    #split data into training-testing sets
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, names, test_size=0.2, random_state=42)

    #train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #make the predictions
    y_pred = model.predict(X_test)

    #evaluate the model's accuracy
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

    test_results = pd.DataFrame({
        'Name': names_test,
        'Actual Points': y_test,
        'Predicted Points': y_pred
    })

    test_results = test_results.reset_index(drop=True)
    print(test_results.head())

else:
    print("Failed to fetch data from FPL API")
