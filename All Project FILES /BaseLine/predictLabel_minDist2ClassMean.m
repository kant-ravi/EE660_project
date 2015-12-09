function predictedLabel=predictLabel_minDist2ClassMean(features,ClassMeans)
% Prediction 
%compute distance of each sample from
%the means and assign the sample to class with least mean
[nSamples,~]=size(features);
predictedLabel=zeros(nSamples,1);
nClasses=size(ClassMeans,1);
for i=1:nSamples
    temp=repmat(features(i,:),nClasses,1);
    temp=temp-ClassMeans;
    temp=sum(temp.^2,2);
    temp=temp.^0.5;
    [~,minIndex]=min(temp);
    if minIndex==nClasses
        predictedLabel(i,1)=0;
    else 
        predictedLabel(i,1)=minIndex; 
    end
        
end
end