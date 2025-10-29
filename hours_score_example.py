# Linear Regression Example: Hours Studied vs Test Score
# Using the intuitive approach you described

# Data points
hours = [1, 2, 3, 4, 5, 6]
scores = [50, 55, 65, 70, 80, 85]

print("Hours Studied (x) -> Test Score (y)")
for h, s in zip(hours, scores):
    print(f"{h} -> {s}")

print("\nCalculating differences between consecutive points:")
differences = []
for i in range(1, len(scores)):
    diff = scores[i] - scores[i-1]
    differences.append(diff)
    print(f"From {hours[i-1]}h to {hours[i]}h: {scores[i-1]} -> {scores[i]} = +{diff} points")

print(f"\nAverage increase per hour: {sum(differences)}/{len(differences)} = {sum(differences)/len(differences):.1f} points")
slope = sum(differences) / len(differences)

# Calculate intercept using one of the points
# Using the first point: Score = b + slope * hours
intercept = scores[0] - slope * hours[0]

print(f"\nUsing point (1, {scores[0]}):")
print(f"{scores[0]} = b + {slope:.1f} * {hours[0]}")
print(f"b = {scores[0]} - {slope:.1f} = {intercept:.1f}")

print(f"\nLinear equation: Score = {intercept:.1f} + {slope:.1f} * Hours")
print("\nVerification:")
for h in hours:
    predicted_score = intercept + slope * h
    print(f"Hours: {h} -> Predicted Score: {predicted_score:.1f}")