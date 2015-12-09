
feature_train = importdata('SIFT_PCA_NaiveBayes_feature_train.txt');
feature_test = importdata('SIFT_PCA_NaiveBayes_feature_test.txt');
label_train = importdata('SIFT_PCA_NaiveBayes_label_train.txt');
label_test = importdata('SIFT_PCA_NaiveBayes_label_test.txt');
train_keypoint_matrix = importdata('SIFT_PCA_NaiveBayes_train_keypoints_matrix.txt');
test_keypoint_matrix = importdata('SIFT_PCA_NaiveBayes_test_keypoints_matrix.txt');

[feature_train,mu,sigma] = zscore(feature_train);
M = repmat(mu,size(feature_test,1),1);
S = repmat(sigma,size(feature_test,1),1);
feature_test = (feature_test - M)./S;
%%
numSamples_trainClass = sum(train_keypoint_matrix,2);
numClasses = size(train_keypoint_matrix,1);
classModels = cell(numClasses,1);
classFeature_startIndex = 1;
for curClass = 1:numClasses
    classFeature_endIndex = classFeature_startIndex + numSamples_trainClass(curClass,1) - 1;
    curTrainingSamples = feature_train(classFeature_startIndex:classFeature_endIndex,:);
    classFeature_startIndex = classFeature_endIndex;
    mdl = KDTreeSearcher(curTrainingSamples);
    classModels{curClass} = mdl;
end

numSamples_testClass  = size(test_keypoint_matrix,2);
numTestImages = numClasses * numSamples_testClass;
classFeature_startIndex = 1;
distanceMatrix=zeros(numTestImages,numClasses);
curRow = 1;
cumIndex = 1;
refTestLabel= zeros(numTestImages,1);
for curClass = 1:numClasses
    for curTestImg = 1:numSamples_testClass
        fprintf('curClass %d curTestImg %d \n',curClass,curTestImg);
        numKeyPoints = test_keypoint_matrix(curClass,curTestImg);
        classFeature_endIndex = classFeature_startIndex + numKeyPoints - 1;
      %  [classFeature_startIndex classFeature_endIndex]
      
        
        curTestImgDescriptor = feature_test(classFeature_startIndex:classFeature_endIndex,:);
       classFeature_startIndex = classFeature_endIndex + 1;
        %  input('Move?');
        for c = 1:numClasses
            mdl = classModels{c};
            [~,D] = knnsearch(mdl,curTestImgDescriptor);
   
            distanceMatrix(cumIndex,c) = sum(D);
          %  fprintf('\t\t Class %d;\n',c);
           % input('next?');
        end
        refTestLabel(cumIndex,1) = curClass;
        cumIndex = cumIndex + 1;
            
    end
    
end
refTestLabel = refTestLabel -1;
[~,predictedLabels] = min(distanceMatrix,[],2); 
predictedLabels = predictedLabels -1;
correct_classification = sum(predictedLabels==refTestLabel)
errorRate = ((numTestImages - correct_classification)/numTestImages) * 100
confusionmat(refTestLabel,predictedLabels)

