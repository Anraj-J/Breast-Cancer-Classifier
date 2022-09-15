from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

training_data, validation_data, training_labels, validation_labels = \
    train_test_split(breast_cancer_data.data,
                     breast_cancer_data.target,
                     test_size=0.2, random_state=100)

accuracies = []
most_accurate = 0
accurate_percentage = 0
least_accurate = 0
least_accurate_percentage = 1
k_list = range(1, 101)

for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    cur = classifier.score(validation_data, validation_labels)
    accuracies.append(cur)

    if cur > accurate_percentage:
        most_accurate = k
        accurate_percentage = cur

    if cur < least_accurate_percentage:
        least_accurate = k
        least_accurate_percentage = cur

print(most_accurate)
print("{:.1%}".format(accurate_percentage))
print(least_accurate)
print("{:.1%}".format(least_accurate_percentage))
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Cancer Classifier Accuracy")
plt.show()
