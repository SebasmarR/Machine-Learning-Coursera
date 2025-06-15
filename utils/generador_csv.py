import random


def linear_regression_csv():
    """
    Generates a CSV file with highly dispersed random data for linear regression.
    The columns 'x' and 'y' will be unrelated and widely varied.
    """
    num_rows = 50
    file = "data/linear_regression.csv"

    with open(file, "w") as f:
        f.write("x,y\n")

        for _ in range(num_rows):
            x = random.uniform(1, 100) + random.gauss(0, 30)
            y = random.uniform(1, 100) + random.gauss(0, 30)
            f.write(f"{x},{y}\n")

    print(f"File created: {file}")


def multiple_linear_regression_csv():
    """
    Generates a CSV file with highly dispersed random data for multiple linear regression.
    No meaningful correlation is present between inputs and output.
    """
    num_rows = 50
    file = "data/multiple_linear_regression_data.csv"

    with open(file, "w") as f:
        f.write("Weight,Volume,CO2\n")

        for _ in range(num_rows):
            weight = random.uniform(1, 100) + random.gauss(0, 30)
            volume = random.uniform(1, 100) + random.gauss(0, 30)

            # CO2 es una mezcla sin relación clara
            co2 = random.uniform(1, 20000) + random.gauss(0, 10000) + \
                random.choice([weight, volume, weight * volume, 0])
            f.write(f"{weight},{volume},{co2}\n")

    print(f"File created: {file}")
