import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("actual.txt", "r") as actual_file, open("predicted.txt", "r") as predicted_file:
    actual_lines = actual_file.readlines()
    predicted_lines = predicted_file.readlines()

def remove_boundary_values(line):
    return line.strip().split()[1:-1]

def plot_confusion_matrix(confusion_mat, num_classes, save_path="confusionMatrix"):
    """
    Plot confusion matrix using seaborn heatmap.

    Args:
    confusion_mat (np.ndarray): Confusion matrix.
    num_classes (int): Number of classes.
    save_path (str): Path to save the plot. If None, the plot will not be saved.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, cmap="Blues", fmt='d', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)  # Save the plot if save_path is provided
    plt.show()

mapper = {
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5,
    '6':6
}
confusionMatrix = [[0 for _ in range(7)] for _ in range(7)]

for actual_line, predicted_line in zip(actual_lines, predicted_lines):
    actual_values = remove_boundary_values(actual_line)
    predicted_values = predicted_line.strip().split()
    for i,j in zip(actual_values,predicted_values):
        confusionMatrix[mapper[i]][mapper[j]]+=1

# Delete the last row
confusion_matrix = confusionMatrix[:-1]

# Delete the last column from each row
confusion_matrix = [row[:-1] for row in confusion_matrix]

for row in confusion_matrix:
    print(row)

plot_confusion_matrix(confusionMatrix,7,"Confusion_Matrix/confusionMatrix7")