clc
clear
feature = importdata('training_feature_histogram_images.txt');
label = importdata('training_label_histogram_images.txt');

ClassMean=minDist2ClassMean(feature,label);

feature = importdata('testing_feature_histogram_images.txt');
label = importdata('testing_labels_histogram_images.txt');

predictedLabel=predictLabel_minDist2ClassMean(feature,ClassMean);
[error,correct]=computError_minDist2ClassMean(label,predictedLabel);
error = error * 100;
correct = correct * 100;
fprintf('\nBase Performance using minimum distance to class mean \n');
fprintf('Error Rate: %.2f\n',error);
fprintf('Correct Rate: %.2f\n',correct);
