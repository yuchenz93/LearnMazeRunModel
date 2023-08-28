function [rwdnow,ccollect,lcollect,rcollect,ridx] = ActRewardPolicy_Dynamic(rwdnow,posnow,npos,rp,ccollect,lcollect,rcollect)
% ActRewardPolicy_Dynamic gives the rewards based on reward policy and past
% experience, we use dynamic programming to speed up
% inputs:rwdnow   reward across bins at previous step
%        posnow   current animal position
%        npos     total number of position bins
%        rp       reward policy, such as 'TLeftRight'
%        ccollect if animal recently collected center reward
%        lcollect if animal recently collected left reward 
%        rcollect if animal recently collected right reward
%        for linear maze, ccollect doesn't matter, we use left and right to
%        represent track start and end
% output:rwdnow  reward value on all position bins at current state
%         ridx   if a reward was collected at current step
%% for linear maze
if strcmp(rp,'altends')
    ridx = 0;
    if  ~lcollect && ~rcollect && posnow == 1
        % animal first time collect reward at start (left)
        rwdnow = zeros(size(rwdnow));
        rwdnow(npos) = 1; % put reward at right
        lcollect = 1;
        rcollect = 0;
        ridx = 1;
    end
    
    if  rwdnow(npos) == 1 && posnow == npos
        % animal collect right reward
        rwdnow = zeros(size(rwdnow));
        rwdnow(1) = 1; % put reward at left
        rcollect = 1;
        lcollect = 0;
        ridx = 1;
    end
    
    if  rwdnow(1) == 1 && posnow == 1
        % animal recenlty collect left reward
        rwdnow = zeros(size(rwdnow));
        rwdnow(npos) = 1; % put reward at right
        lcollect = 1;
        rcollect = 0;
        ridx = 1;
    end
end

%% for T maze
% for T maze, if we have npos position bins, then we will have floor(npos/3)
% bins for left and right arms, and remaining are center arms
% we will organize bins as center-left-right
if strcmp(rp,'TLeftRight')
    ridx = 0;
    nright = floor(npos/3);
    nleft = nright;
    ncenter = npos - nleft - nright;
    % alternative rewards placed at left and right arms, then reset at
    % middle arm
    
    
    if ~ccollect && ~lcollect && ~rcollect && posnow == 1
        % animal first time collect center reward
        rwdnow = zeros(size(rwdnow));
        rwdnow(ncenter+nleft) = 1; % put reward at left
        ccollect = 1;
        ridx = 1;
    end
    
    if ~ccollect && ~lcollect && rcollect && posnow == 1
        % animal recenlty collect right reward
        rwdnow = zeros(size(rwdnow));
        rwdnow(ncenter+nleft) = 1; % put reward at left
        ccollect = 1;
        ridx = 1;
    end
    
    if ~ccollect && lcollect && ~rcollect && posnow == 1
        % animal recenlty collect left reward
        rwdnow = zeros(size(rwdnow));
        rwdnow(npos) = 1; % put reward at right
        ccollect = 1;
        ridx = 1;
    end
    
    if  rwdnow(ncenter+nleft) == 1 && posnow == ncenter+nleft
        % animal collect left reward
        rwdnow = zeros(size(rwdnow));
        rwdnow(1) = 1; % put reward back to center
        ccollect = 0;
        lcollect = 1;
        rcollect = 0;
        ridx = 1;
    end
    
    if  rwdnow(npos) == 1 && posnow == npos
        % animal collect right reward
        rwdnow = zeros(size(rwdnow));
        rwdnow(1) = 1; % put reward back to center
        ccollect = 0;
        lcollect = 0;
        rcollect = 1;
        ridx = 1;
    end
    
end
end

