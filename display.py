from controller import *


def main():
    """Main function to run the regression analysis tool."""
    while True:
        print("\nRegression Analysis Tool:")
        print("1. Linear Regression\n")
        print("2. Multiple Linear Regression\n")
        print("3. Polynomial Regression\n")
        print("4. Logistic Regression\n")
        print("5. Exit\n")

        choice = input("Enter your choice: ")

        options = {
            "1": linear_regression_func,
            "2": multiple_linear_regression_func,
            "3": polynomial_regression_func,
            "4": logistic_regression_func
        }

        if choice == "5":
            print("\nExiting...")
            break

        func = options.get(choice)
        if func:
            func()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
