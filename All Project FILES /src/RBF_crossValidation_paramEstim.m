clear
clc
feature = importdata('training_feature_histogram_images.txt');
label = importdata('training_label_histogram_images.txt');
fprintf('\nTraining data loaded')
%% Normalize
feature=zscore(feature);
[nSamples,nFeatures]=size(feature);
nClasses=size(unique(label),1);
fprintf('\n Normalizing Done')
%% data has 30 samples of each class 

numFolds=5;
numSubFolds=2;

foldSize=6; %numSamplesPerFold
numSamplesPerDigitPerFold=2;


curFold=1;

 %%    
 fprintf('\n Starting Cross Validation and Parameter estimation')
 fprintf('\n Working...')
curTestFold=1;  
grid=[];
bestGrid=[];
foldTestAccuracy_matrix=[];
Opt_C=0;
Opt_G=0;
Max_Accuracy=0;
while curTestFold<=numFolds
    curTestFold_feature=feature(1+((curTestFold-1)*foldSize):(curTestFold*foldSize),:);
    curTestFold_label=label(1+((curTestFold-1)*foldSize):(curTestFold*foldSize),1);
    
    curTrainingFolds_feature=[feature(1:(curTestFold-1)*foldSize,:);feature(curTestFold*foldSize+1:end,:)];
    curTrainingFolds_label=[label(1:(curTestFold-1)*foldSize,1);label(curTestFold*foldSize+1:end,1)];  
    
    curValidSubFold=1;
    bestValid_Accuracy=0;
    bestValid_c=0;
    bestValid_g=0;
 %   input 'check 1, proceed?'
    fprintf('\n*******************************************')
    fprintf('\nCurrent Test Fold:%d',curTestFold);
    fprintf('\n*******************************************')
    while curValidSubFold<=numSubFolds
        fprintf('\n=============================================')
        fprintf('\nCurrent Validation Test SubFold:%d',curValidSubFold);
        fprintf('\n=============================================')
        curValid_TestFeature=curTrainingFolds_feature(1+((curValidSubFold-1)*foldSize):(curValidSubFold*foldSize),:);
        curValid_TestLabel=curTrainingFolds_label(1+((curValidSubFold-1)*foldSize):(curValidSubFold*foldSize),1);
        
        curValid_TrainingFeature=[curTrainingFolds_feature(1:(curValidSubFold-1)*foldSize,:);curTrainingFolds_feature(curValidSubFold*foldSize+1:end,:)];
        curValid_TrainingLabel=[curTrainingFolds_label(1:(curValidSubFold-1)*foldSize,1);curTrainingFolds_label(curValidSubFold*foldSize+1:end,1)];
        
        bestAccuracy = 0;
        bestc=0;
        bestg=0;
   %     input 'check 1.1, proceed?'
        p1=1;
        for log2c = 0:9,
            fprintf('\n--------------------------------')
            fprintf('\n Run %d \n',p1);
            fprintf('--------------------------------\n')
            p2=1;
            accuracy=0;
            for log2g = -10:-3,
                if accuracy<80 && accuracy~=0
                    continue;
                end
                fprintf('\n Run %d_%d \n',p1,p2);
                cmd = ['-c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -q'];
                model = svmtrain(curValid_TrainingLabel, curValid_TrainingFeature, cmd);
                [~, accuracy,~]=svmpredict(curValid_TestLabel,curValid_TestFeature,model);
                accuracy=accuracy(1,1);
                grid=[grid;accuracy 2^log2c 2^log2g];
                
                if (accuracy >= bestAccuracy),
                  bestAccuracy = accuracy; bestc = 2^log2c; bestg = 2^log2g;
                end
                fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, accuracy, bestc, bestg, bestAccuracy);
               
                p2=p2+1;
                
           end
        p1=p1+1;  
        end
        bestGrid=[bestGrid;bestAccuracy,bestc,bestg]; 
        if bestAccuracy>=bestValid_Accuracy
           bestValid_Accuracy= bestAccuracy;
           bestValid_c=bestc;
           bestValid_g=bestg;
        end
%        input 'check 2, proceed?'
        curValidSubFold=curValidSubFold+1;
    end
    
    % test on curTestFold
    cmd = ['-c ', num2str(bestValid_c), ' -g ', num2str(bestValid_g),' -q'];
    model = svmtrain(curTrainingFolds_label,curTrainingFolds_feature, cmd);
    [~, accuracy,~]=svmpredict(curTestFold_label,curTestFold_feature,model);
    accuracy=accuracy(1,1);
    foldTestAccuracy_matrix=[foldTestAccuracy_matrix;accuracy,bestValid_c,bestValid_g];
    if accuracy>=Max_Accuracy
        Max_Accuracy=accuracy;
        Opt_C=bestValid_c;
        Opt_G=bestValid_g;
    end
    curTestFold=curTestFold+1;
%    input 'check 3, proceed?'
end
fprintf('\n Cross Validation and Parameter Estimation done.')
%input 'check 4, proceed?'
estimatedAccuracy=sum(foldTestAccuracy_matrix(:,1))/size(foldTestAccuracy_matrix,1);
fprintf('\n The estimated error is : %.3f',estimatedAccuracy)
fprintf('\n Optimal C = %.4f',Opt_C)
fprintf('\n Optimal G = %.6f',Opt_G)
%input 'check 5, proceed?'
%%
fprintf('\n Now training on entire data set...\n')
feature = importdata('training_feature_histogram_images.txt');
label = importdata('training_label_histogram_images.txt');
feature=zscore(feature);
cmd = ['-c ', num2str(Opt_C), ' -g ', num2str(Opt_G),' -q'];
model = svmtrain(label, feature, cmd);

feature = importdata('testing_feature_histogram_images.txt');
label = importdata('testing_labels_histogram_images.txt');

feature=zscore(feature);
fprintf('\nTesting the model\n')
[predicted_label, accuracy,~]=svmpredict(label,feature,model);
fprintf('\n Accuracy on Test Data : %.3f',accuracy(1,1));
fprintf('\n\n----DONE----\n');
        