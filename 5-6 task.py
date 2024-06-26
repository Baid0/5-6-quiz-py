import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    def clean_numeric(value):
        if isinstance(value, str):
            return float(value.replace(',', '').replace('$', ''))
        return float(value)

    numeric_columns = ['AveragePlayerAge', 'TotalGoalsLastSeason', 'MatchesWonLastSeason',
                       'MatchesDrawnLastSeason', 'MatchesLostLastSeason', 'TotalGoalsConcededLastSeason',
                       'TotalRevenueLastSeason', 'StadiumCapacity', 'AverageAttendance',
                       'TransferSpendingLastSeason', 'TransferIncomeLastSeason', 'NumberOfTrophies',
                       'MarketValueOfSquad', 'AveragePlayerMarketValue', 'YouthAcademyRating', 'Price']

    for col in numeric_columns:
        df[col] = df[col].apply(clean_numeric)

    return df

# 1. Simple Linear Regression
def simple_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

# 2. Multiple Linear Regression
def multiple_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

# 3. გადაწყვეტილების ხის რეგრესიის მოდელი
def decision_tree_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

# 4. ლოგისტიკური რეგრესიის მოდელი
def logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report

# 5. გადაწყვეტილების ხის კლასიფიკაცია
def decision_tree_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report


def main():
    df = load_and_clean_data('Football_teams_price_data.csv')

    # 1. Simple Linear Regression
    X_simple = df[['MarketValueOfSquad']].values
    y = df['Price'].values
    simple_model, simple_mse, simple_r2 = simple_linear_regression(X_simple, y)
    print("1. Simple Linear Regression Results:")
    print(f"Mean Squared Error: {simple_mse}")
    print(f"R-squared Score: {simple_r2}")

    # 2. Multiple Linear Regression
    X_multiple = df[['MarketValueOfSquad', 'TotalRevenueLastSeason', 'StadiumCapacity', 'AverageAttendance']].values
    multiple_model, multiple_mse, multiple_r2 = multiple_linear_regression(X_multiple, y)
    print("\n2. Multiple Linear Regression Results:")
    print(f"Mean Squared Error: {multiple_mse}")
    print(f"R-squared Score: {multiple_r2}")

    # 3. გადაწყვეტილების ხის რეგრესია
    dt_reg_model, dt_reg_mse, dt_reg_r2 = decision_tree_regression(X_multiple, y)
    print("\n3. Decision Tree Regression Results:")
    print(f"Mean Squared Error: {dt_reg_mse}")
    print(f"R-squared Score: {dt_reg_r2}")

    # 4. ლოგისტიკური რეგრესია
    df['HighPrice'] = (df['Price'] > df['Price'].median()).astype(int)
    y_class = df['HighPrice'].values
    log_model, log_accuracy, log_report = logistic_regression(X_multiple, y_class)
    print("\n4. Logistic Regression Results:")
    print(f"Accuracy: {log_accuracy}")
    print("Classification Report:")
    print(log_report)

    # 5. გადაწყვეტილების ხის კლასიფიკაცია
    dt_class_model, dt_class_accuracy, dt_class_report = decision_tree_classification(X_multiple, y_class)
    print("\n5. Decision Tree Classification Results:")
    print(f"Accuracy: {dt_class_accuracy}")
    print("Classification Report:")
    print(dt_class_report)

if __name__ == "__main__":
    main()