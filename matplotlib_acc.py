import matplotlib.pyplot as plt

# Accuracy data
raw_feature_euclidean = [0.3567, 0.3417, 0.2811]
raw_feature_cosine = [0.3567, 0.3417, 0.2794]
edge_feature_euclidean = [0.254, 0.2348,  0.239]
edge_feature_cosine = [0.254, 0.2348, 0.233]
cnn = [0.9491]

labels = ['NN', 'KNN', 'SVM']

# Create line graph
plt.figure(figsize=(10,5))
plt.plot(labels, raw_feature_euclidean, marker='o', label='Raw Feature Euclidean')
plt.plot(labels, raw_feature_cosine, marker='o', label='Raw Feature Cosine')
plt.plot(labels, edge_feature_euclidean, marker='o', label='Edge Feature Euclidean')
plt.plot(labels, edge_feature_cosine, marker='o', label='Edge Feature Cosine')

# Add accuracy values to the plot
for i, acc in enumerate(raw_feature_euclidean):
    plt.text(i, acc, f' {acc}', verticalalignment='bottom')
for i, acc in enumerate(raw_feature_cosine):
    plt.text(i, acc, f' {acc}', verticalalignment='top')
for i, acc in enumerate(edge_feature_euclidean):
    plt.text(i, acc, f' {acc}', verticalalignment='bottom')
for i, acc in enumerate(edge_feature_cosine):
    plt.text(i, acc, f' {acc}', verticalalignment='top')

plt.legend()

# Add CNN accuracy
plt.scatter(['CNN'], cnn, marker='o', color='r')
plt.text('CNN', cnn[0], f' {cnn[0]}', verticalalignment='bottom')

plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.grid(True)
plt.savefig(f'acc_all.png')
