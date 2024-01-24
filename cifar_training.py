import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from skimage.feature import hog
from skimage.filters import sobel

# Download and save CIFAR10 data without normalization
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# Convert the data to NumPy arrays without normalization
train_images = np.array(train_data.data)
train_labels = np.array(train_data.targets)

test_images = np.array(test_data.data)
test_labels = np.array(test_data.targets)

# Define global variables for DL 
batch_size = 32
image_size = 224
num_epochs = 50
dl_task_name="resnet18"
learning_rate=0.001
# Download and save CIFAR10 data with normalization for CNN
transform_dl = transforms.Compose([
    transforms.Resize((image_size, image_size)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_data_dl = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_dl)
test_data_dl = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_dl)

train_loaders = DataLoader(train_data_dl, batch_size=batch_size, shuffle=True)
test_loaders = DataLoader(test_data_dl, batch_size=batch_size, shuffle=False)

# Function to visualize class distribution
def visualize_class_distribution(train_dataset, test_dataset, title):
    class_counts_train = np.zeros(10)
    class_counts_test = np.zeros(10)
    for _, label in train_dataset:
        class_counts_train[label] += 1
    for _, label in test_dataset:
        class_counts_test[label] += 1

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    x = np.arange(len(classes))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, class_counts_train, width, label='Train')
    rects2 = ax.bar(x + width/2, class_counts_test, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()

    fig.tight_layout()
    plt.savefig('cifar10_dataset.png')

# 创建全局的图形对象
pr_fig, pr_ax = plt.subplots()
roc_fig, roc_ax = plt.subplots()

# Define function to compute metrics and plot curves
def compute_metrics_and_plot_curves(all_labels, all_predictions, task_name=""):
    # Compute precision, recall, PR curve, and ROC curve here...
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions, pos_label=1)
    fpr, tpr, _ = roc_curve(all_labels, all_predictions, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 在全局的PR图形对象上添加新的曲线
    pr_ax.plot(recall, precision, lw=2, label=task_name)

    # 在全局的ROC图形对象上添加新的曲线
    roc_ax.plot(fpr, tpr, lw=2, label=f'{task_name} (area = {roc_auc:.2f})')

def train_and_evaluate_with_shallow_methods(train_images, test_images):
    # Define the feature types, distance types and classifiers
    feature_types = ["raw", "edge"]
    distance_types = ["euclidean", "cosine"]
    classifiers = {"NN": KNeighborsClassifier(n_neighbors=1), "KNN": KNeighborsClassifier(), "SVM": LinearSVC()}

    # Iterate over all combinations of feature types, distance types and classifiers
    for feature_type in feature_types:
        for distance_type in distance_types:
            for classifier_name, classifier in classifiers.items():
                # Choose the feature type
                if feature_type == "raw":
                    X_train = train_images.reshape(len(train_images), -1)
                    X_test = test_images.reshape(len(test_images), -1)
                elif feature_type == "edge":
                    X_train = np.array([sobel(img) for img in train_images]).reshape(len(train_images), -1)
                    X_test = np.array([sobel(img) for img in test_images]).reshape(len(test_images), -1)

                y_train = train_labels
                y_test = test_labels

                # Normalize the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Training code here...
                classifier.fit(X_train, y_train)

                # Evaluation code here...
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f'Accuracy of the {feature_type} feature, {distance_type} distance and {classifier_name} on the Test Set: {accuracy}')

                # Compute precision, recall, PR curve and ROC curve here...
                compute_metrics_and_plot_curves(y_test, y_pred, task_name=f"{feature_type}_{distance_type}_{classifier_name}")

# Define function to train and evaluate with ResNet18
def train_and_evaluate_with_cnn(train_loaders, test_loaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_func = getattr(models, dl_task_name)
    model = model_func(weights=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training code here...
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loaders, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Print statistics (accuracy) every epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loaders:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Accuracy on Test Set: {accuracy:.2f}%')

    print('Finished Training')
    
    # Evaluation code here...
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data in test_loaders:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            # Save the predicted probabilities for the positive class
            all_predictions.extend(probabilities[:, 1].cpu().numpy())

    print(f'Final Accuracy on the Test Set: {accuracy:.2f}%')

    compute_metrics_and_plot_curves(all_labels, all_predictions, task_name=dl_task_name)

def main():
    # Visualize class distribution in the training and testing datasets
    visualize_class_distribution(train_data_dl, test_data_dl, title='Class Distribution in Datasets')
    
    # Train and evaluate with  shallow methods
    print("\nTraining and evaluating with shallow methods(NN, KNN, SVM):")
    train_and_evaluate_with_shallow_methods(train_images, test_images)

    # Train and evaluate with CNN
    print("\nTraining and evaluating with CNN (resnet18):")
    train_and_evaluate_with_cnn(train_loaders, test_loaders)

    # 设置PR图形的属性并保存
    pr_ax.set_xlim([0.0, 1.0])
    pr_ax.set_ylim([0.0, 1.05])
    pr_ax.set_xlabel('Recall')
    pr_ax.set_ylabel('Precision')
    pr_ax.set_title('Precision-Recall curve')
    pr_ax.legend(loc="lower right")
    pr_fig.savefig('PR_curve.png')

    # 设置ROC图形的属性并保存
    roc_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    roc_ax.set_xlim([0.0, 1.0])
    roc_ax.set_ylim([0.0, 1.05])
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('Receiver Operating Characteristic')
    roc_ax.legend(loc="lower right")
    roc_fig.savefig('ROC_curve.png')

if __name__ == "__main__":
    main()