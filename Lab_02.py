# Sample dataset (features: weather, time, experience; label: gym attendance)
data = [
    ['rainy', 'morning', 'beginner', 'no'],
    ['rainy', 'evening', 'intermediate', 'yes'],
    ['sunny', 'morning', 'advanced', 'yes'],
    ['sunny', 'afternoon', 'beginner', 'no'],
    ['cloudy', 'evening', 'beginner', 'yes'],
    ['cloudy', 'morning', 'intermediate', 'yes'],
    ['rainy', 'afternoon', 'advanced', 'no'],
    ['sunny', 'evening', 'intermediate', 'yes'],
    ['sunny', 'morning', 'beginner', 'yes'],
    ['cloudy', 'afternoon', 'advanced', 'yes']
]

# Separate inputs and labels
features = [entry[:-1] for entry in data]
labels = [entry[-1] for entry in data]

# Compute prior probabilities
def compute_prior(labels):
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    total_labels = len(labels)
    return {label: count / total_labels for label, count in label_counts.items()}

# Compute likelihood probabilities
def compute_likelihood(features, labels):
    likelihoods = {}
    feature_count = len(features[0])
    
    for i in range(feature_count):
        likelihoods[i] = {}
        for label in set(labels):
            likelihoods[i][label] = {}
            filtered_features = [features[j] for j in range(len(features)) if labels[j] == label]
            feature_values = [item[i] for item in filtered_features]
            for value in set(feature_values):
                likelihoods[i][label][value] = feature_values.count(value) / len(feature_values)
    return likelihoods

# Compute posterior probability
def compute_posterior(sample, prior, likelihood):
    posterior = {}
    for label in prior:
        probability = prior[label]
        for i in range(len(sample)):
            value = sample[i]
            if value in likelihood[i][label]:
                probability *= likelihood[i][label][value]
            else:
                probability *= 0
        posterior[label] = probability
    return posterior

# Prediction function
def classify(sample, prior, likelihood):
    probabilities = compute_posterior(sample, prior, likelihood)
    return max(probabilities, key=probabilities.get)

# Train model
prior_probabilities = compute_prior(labels)
likelihood_probabilities = compute_likelihood(features, labels)

# New sample to predict
new_entry = ['sunny', 'afternoon', 'intermediate']
predicted_result = classify(new_entry, prior_probabilities, likelihood_probabilities)
print(f"The predicted gym attendance for {new_entry} is: {predicted_result}")
