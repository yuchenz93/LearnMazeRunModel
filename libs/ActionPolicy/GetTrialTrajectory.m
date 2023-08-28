function tra = GetTrialTrajectory(act)
% this function extract the full trajectory from a trial, which is just the
% first element from all the state varibles with memory

tra = nan(1,length(act));
for iseg = 1:length(act)
    tra(iseg) = act{iseg}(1);
end

end