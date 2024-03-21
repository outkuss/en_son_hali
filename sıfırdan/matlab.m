% Load the training data
load('training_data.mat');

% Load the training labels
load('training_labels.mat');

% Initialize the k-NN classifier
knn = fitcknn(training_data', training_labels', 'k', 1);

% Load the test data
load('test_data.mat');

% Classify the test data using the k-NN classifier
test_labels = predict(knn, test_data');

% Print the accuracy of the classifier
accuracy = sum(test_labels == test_labels_true) / numel(test_labels_true);
disp(['Accuracy: ', num2str(accuracy)]);

% Classify a single music file
file_features = extract_features('path/to/music/file.mp3');
file_features = reshape(file_features, [1, numel(file_features)]);
label = predict(knn, file_features);
disp(['This music is most likely a ', training_labels(label)]);