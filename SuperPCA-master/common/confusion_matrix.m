function [confusion, accuracy, TPR, FPR,oa,aa,kp] = confusion_matrix(class, c)
%
% class is the result of test data after classification
%          (1 x n)
%
% c is the label for testing data
%          (1 x len_c)
%
%

class = class';
c = c.';

n = length(class);
c_len = length(c);

if n ~= sum(c)
    disp('WRANING:  wrong inputting!');
    return;
end


% confusion matrix
confusion = zeros(c_len, c_len);
a = 0;
for i = 1: c_len
    for j = (a + 1): (a + c(i))
        confusion(i, class(j)) = confusion(i, class(j)) + 1;
    end
    a = a + c(i);
end


% True_positive_rate + False_positive_rate + accuracy
TPR = zeros(1, c_len);
FPR = zeros(1, c_len);
oa=0;
for i = 1: c_len
  FPR(i) = confusion(i, i)/sum(confusion(:, i));
  TPR(i) = confusion(i, i)/sum(confusion(i, :));
  oa=oa+confusion(i, i);
end
oa=oa/sum(sum(confusion));
aa=sum(TPR)/length(TPR);
ka=sum(confusion);
kb=sum(confusion');
N=sum(ka);
pe=sum(ka.*kb/N^2);
po=oa;
kp=(po-pe)/(1-pe);
accuracy = sum(diag(confusion))/sum(c);
