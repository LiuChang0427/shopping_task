import csv
import sys
import calendar
import math

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.4

file_path = "/Users/mitaonaicha/Desktop/UniversityofKlagenfurt_Task/shopping/shopping.csv"

def main():

    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = r"/Users/mitaonaicha/Desktop/UniversityofKlagenfurt_Task/shopping/shopping.csv"

    evidence, labels = load_data(file)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions using the nearest neighbor classifier
    predictions = nearest_neighbor(X_train, y_train, X_test)
    sensitivity, specificity, F1 = evaluate(y_test, predictions)

    print(f"Correct: {sum(1 for true, pred in zip(y_test, predictions) if true == pred)}")
    print(f"Incorrect: {sum(1 for true, pred in zip(y_test, predictions) if true != pred)}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"F1 Score: {F1:.2f}")



# data preprocessing
def load_data(file):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    evidence = []
    labels = []

    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return evidence, labels


# train & predict
def nearest_neighbor(evidence, labels, test_evidence):
    predictions = []

    for test_point in test_evidence:
        min_distance = math.inf
        predicted_label = None

        for i, train_point in enumerate(evidence):
            distance = euclidean_distance(test_point, train_point)
            if distance < min_distance:
                min_distance = distance
                predicted_label = labels[i]

        predictions.append(predicted_label)

    return predictions



# Euclidean distance
def euclidean_distance(point1, point2):
    # Calculate Euclidean distance between two points
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))


# Calculate of precision & sensitivity
def evaluate (labels, predictions):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for label, prediction in zip(labels, predictions):
        if label == 1:
            if prediction == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if prediction == 0:
                true_negatives += 1
            else:
                false_positives += 1

    if true_positives + false_positives == 0:
        precision = 0

    elif true_positives + false_negatives == 0:
        recall = 0
    else:
    # Sensitivity (True Positive Rate) 敏感度
        sensitivity = true_positives / (true_positives + false_negatives)

    # Specificity (True Negative Rate) 特异性
        specificity = true_negatives / (true_negatives + false_positives)
    #  Precision & recall
        precision = true_positives / (true_positives + false_positives)
        recall = sensitivity

    # F1 Measure
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)

    return sensitivity, specificity, F1



if __name__ == "__main__":
    main()
