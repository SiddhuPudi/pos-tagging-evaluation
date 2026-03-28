import matplotlib.pyplot as plt
def plot_accuracy(models, accuracies):
    plt.figure()
    plt.bar(models, accuracies)
    plt.title("Model Accuracy Comparsion")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.show()