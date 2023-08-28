function [TAct,TR] = GetTrainingSetfromExp(Act,npos,rp)
% this function get traning sets from experience 
% inputs:Act,  is a behavior set, is a cell array contians behavior data from
%              several trails. Within each trail, it's actual a continuous
%              experience with transition states (linear positionS),
%              however, it was cutted into many cells, with each cell being
%              a segment of trajectory with finite memory length.
%              for example, in a trial, if the animal has a trajectory 
%              1-2-3-4-5-..20, and if the memory length is 5, we will cut
%              it into {1,2,3,4,5},{2,3,4,5,6},...{16,17,18,19,20}
%        npos, total number of position bins
%        rp,   actual reward policy
% outputs: TAct, is a matrix with columns with realizations, rows being
%                history of behavior states
%          TR,   is a matrix with columns with realizations, rows being
%                reward across spatial bins for each sample,
% the row and column in TAct and TR are arranged in this way to match the
% default Matlab fitting neural network inputs and targets

TAct = [];
TR = [];
for it = 1:length(Act)
    for iseg = 1:length(Act{it})
        seqnow = Act{it}{iseg};
        % convert it to column vector
        if isrow(seqnow)
            seqnow = seqnow';
        end
        % get actual reward based on reward policy
        rwdnow = ActRewardPolicy(seqnow,npos,rp);
        if isrow(rwdnow)
            rwdnow = rwdnow';
        end
        
        TAct = cat(2,TAct,seqnow);
        TR = cat(2,TR,rwdnow);
        
    end
end



end