import pandas
import logic.multiple_linear_regression as mlr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from utils.generador_csv import multiple_linear_regression_csv


multiple_linear_regression_csv()

df = pandas.read_csv("data/multiple_linear_regression_data.csv")

# Using the sklearn multiple linear regression module

x_df = df[['Weight', 'Volume']]
y_df = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(x_df, y_df)
y_pred = regr.predict(x_df)

print("Slopes (w):", regr.coef_)
print("Intercept (b):", regr.intercept_)

print(f"\nFunction of the model:")
print(
    f"CO2 = ({regr.coef_[0]:.2f} * Weight) + ({regr.coef_[1]:.2f} * Volume) + {regr.intercept_:.2f}")

print("\nMean square error (MSE):", mean_squared_error(y_df, y_pred))

# Using the coursera multiple linear regression module

x = x_df.values.tolist()
y = y_df.values.tolist()

w, b, error = mlr.gradient_descent(x, y, 0.01, 10000)
print("\nCoursera Multiple Linear Regression:")
print("Slopes (w):", w)
print("Intercept (b):", b)
print(f"\nFunction of the model:")
print(f"CO2 = ({w[0]:.2f} * Weight) + ({w[1]:.2f} * Volume) + {b:.2f}")
print("\nMean square error (MSE):", error)

# Using the custom multiple linear regression module

w, b, error = mlr.multiple_linear_regression(x, y)
print("\nCustom Multiple Linear Regression:")
print("Slopes (w):", w)
print("Intercept (b):", b)
print(f"\nFunction of the model:")
print(f"CO2 = ({w[0]:.2f} * Weight) + ({w[1]:.2f} * Volume) + {b:.2f}")
print("\nMean square error (MSE):", error)
