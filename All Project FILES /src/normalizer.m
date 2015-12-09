
feature_train = importdata('training_feature_histogram_images.txt');
feature_test = importdata('testing_feature_histogram_images.txt');
%%
[feature_train,mu,sigma] = zscore(feature_train);
M = repmat(mu,size(feature_test,1),1);
S = repmat(sigma,size(feature_test,1),1);
feature_test = (feature_test - M)./S;
%%
dlmwrite('training_feature_histogram_images.txt',feature_train,'delimiter',' ','precision',5);
dlmwrite('testing_feature_histogram_images.txt',feature_test,'delimiter',' ','precision',5);