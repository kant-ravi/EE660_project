function [errorRate,correctRate]=computError_minDist2ClassMean(label,predicted_label)
correct=0;
error=0;
nSamples=size(label,1);
for i=1:nSamples
    if predicted_label(i,1)==label(i,1)
        correct=correct+1;
    else
        error=error+1;
    end
end
errorRate=error/nSamples;
correctRate=correct/nSamples;
end