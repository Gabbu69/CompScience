import numpy as np
import matplotlib.pyplot as plt

# Step 1: Ask the user to input height and weight data
print("Enter height and weight data (e.g., '170 70'). Type 'done' when finished:")
heights = []  # To store the height values
weights = []  # To store the weight values

# Collect data until the user types 'done'
while True:
    user_input = input("Height and Weight: ").strip()
    if user_input.lower() == "done":
        break  # Exit the loop if the user is done
    try:
        # Try to split the input into two float values (height and weight)
        height, weight = map(float, user_input.split())
        heights.append(height)  # Add height to the list
        weights.append(weight)  # Add weight to the list
    except ValueError:
        # If the input is not valid, show an error message
        print("Invalid input. Please enter two numbers separated by a space.")

# Step 2: Convert the height and weight data into numpy arrays
x = np.array(heights)  # Heights as x values
y = np.array(weights)  # Weights as y values

# Step 3: Check if we have enough data for the regression
if len(x) < 2:
    print("Not enough data to perform regression. Please input at least two data points.")
    exit()  # Exit if there are not enough data points

# Step 4: Calculate the mean (average) of the height and weight
mean_x = np.mean(x)  # Mean of the heights
mean_y = np.mean(y)  # Mean of the weights

# Step 5: Calculate the slope (m) and intercept (b) for the regression line
numerator = np.sum((x - mean_x) * (y - mean_y))  # Top part of the slope formula
denominator = np.sum((x - mean_x) ** 2)          # Bottom part of the slope formula
m = numerator / denominator                      # The slope (m)
b = mean_y - m * mean_x                          # The intercept (b)

# Step 6: Use the regression line to predict weight values based on height
y_pred = m * x + b  # Predicted weight values using the regression line

# Step 7: Calculate the Mean Squared Error (MSE) to evaluate the model's accuracy
mse = np.mean((y - y_pred) ** 2)  # Average squared difference between actual and predicted weights

# Step 8: Calculate the R-squared value to measure the goodness of fit
ss_total = np.sum((y - mean_y) ** 2)  # Total variation in weight
ss_residual = np.sum((y - y_pred) ** 2)  # Remaining variation after regression
r_squared = 1 - (ss_residual / ss_total)  # How well the model fits the data

# Step 9: Output the results
print("\nLinear Regression Results")
print(f"Equation: y = {m:.2f}x + {b:.2f}")  # Display the regression equation
print(f"Mean Squared Error (MSE): {mse:.2f}")  # Display MSE
print(f"R-squared: {r_squared:.2f}")  # Display R-squared

# Step 10: Plot the original data and the regression line
plt.scatter(x, y, color="blue", label="Data Points")         # Plot original data points as blue dots
plt.plot(x, y_pred, color="red", label="Regression Line")    # Plot regression line in red
plt.xlabel("Height (e.g., cm)")  # Label for the x-axis
plt.ylabel("Weight (e.g., kg)")  # Label for the y-axis
plt.title("Linear Regression")  # Title of the plot
plt.legend()  # Show legend
plt.grid(True)  # Show grid for better readability
plt.show()  # Display the plot
