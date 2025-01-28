import numpy as np
import matplotlib.pyplot as plt

# Prompt user for height and weight data
print("Enter height and weight data (e.g., '170 70'). Type 'done' when finished:")
heights = []
weights = []

while True:
    user_input = input("Height and Weight: ").strip()
    if user_input.lower() == "done":
        break
    try:
        height, weight = map(float, user_input.split())
        heights.append(height)
        weights.append(weight)
    except ValueError:
        print("Invalid input. Please enter two numbers separated by a space.")

# Convert input data into numpy arrays
x = np.array(heights)
y = np.array(weights)

# Check if enough data is entered
if len(x) < 2:
    print("Not enough data to perform regression. Please input at least two data points.")
    exit()

# Prior assumptions for the slope and intercept
prior_m = 1  # Initial guess for slope
prior_b = 0  # Initial guess for intercept

# Compute the mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate the slope (m) and intercept (b) from the data (likelihood)
numerator = np.sum((x - mean_x) * (y - mean_y))  # Top part of slope formula
denominator = np.sum((x - mean_x) ** 2)          # Bottom part of slope formula
likelihood_m = numerator / denominator           # Data-driven slope
likelihood_b = mean_y - likelihood_m * mean_x    # Data-driven intercept

# Combine prior beliefs with likelihood (simple averaging)
posterior_m = (prior_m + likelihood_m) / 2  # Final slope
posterior_b = (prior_b + likelihood_b) / 2  # Final intercept

# Predict y values using the posterior slope and intercept
y_pred = posterior_m * x + posterior_b

# Output the Bayesian-inspired regression equation
print("\nBayesian-Inspired Linear Regression")
print(f"Equation: y = {posterior_m:.2f}x + {posterior_b:.2f}")

# Plot the original data points and the regression line
plt.scatter(x, y, color="blue", label="Data Points")         # Actual data points
plt.plot(x, y_pred, color="red", label="Regression Line")    # Fitted line
plt.xlabel("Height (e.g., cm)")
plt.ylabel("Weight (e.g., kg)")
plt.title("Bayesian-Inspired Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
