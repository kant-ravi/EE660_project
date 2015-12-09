function ClassMeans=minDist2ClassMean(features,labels)

%% compute class means
[nSamples,nFeatures]=size(features);
nClasses=size(unique(labels),1);
Count=zeros(nClasses,1);
Summation=zeros(nClasses,nFeatures);
ClassMeans=zeros(nClasses,nFeatures);

for i=1:nSamples
    if labels(i,1)~=0
        Summation(labels(i,1),:)= Summation(labels(i,1),:)+features(i,:);
        Count(labels(i,1),1)=Count(labels(i,1),1)+1;
    else
        Summation(nClasses,:)= Summation(nClasses,:)+features(i,:);
        Count(nClasses,1)=Count(nClasses,1)+1;
    end
end

for i=1:nClasses
    ClassMeans(i,:)=Summation(i,:)/Count(i,1);
end


end
    