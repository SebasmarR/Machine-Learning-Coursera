def read_csv(file):
    """
    Reads a CSV file

    Args:
        file (str): Path to the CSV file.

    Returns:
        Tuple: A tuple of two lists, the first containing the list of x values and the second containing the list of y values.
    """

    x = []
    y = []

    with open(file, 'r') as file:
        next(file)
        line = file.readline()

        while line:
            line_parts = line.strip().split(',')

            if len(line_parts) >= 2:
                try:
                    x_value = float(line_parts[0])
                    y_value = float(line_parts[1])
                    x.append(x_value)
                    y.append(y_value)
                except ValueError:
                    print(
                        f"Skipping line due to conversion error: {line.strip()}")

            else:
                print(
                    f"Skipping line due to insufficient data: {line.strip()}")
            line = file.readline()

    return x, y
