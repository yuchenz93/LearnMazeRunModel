function rwdnow = ActRewardPolicy(Sh,npos,rp)
% ActRewardPolicy gives the rewards based on reward policy and past
% experience
% inputs:Sh   vector of state history, starting from current state to previous
%             states, values should be position index,marking the linear position 
%             at each time step
%        npos total number of position bins
%        rp   reward policy, such as 'altends' 
% output:rwdnow  reward value on all position bins at current state

%% for linear maze
if strcmp(rp,'altends')
    % alternative rewards placed at track ends
    behpast = Sh(1:end);
    startrh = find(behpast == 1,1,'first');
    endrh = find(behpast == npos,1,'first');
    
    rwdnow = zeros(1,npos);
    % never visited both ends
    if isempty(startrh) && isempty(endrh)
        rwdnow(1) = 1;
    end
    
    % never visited start but visited end
    if isempty(startrh) && ~isempty(endrh)
        rwdnow(1) = 1;
    end
    
    % never visited end but visited start
    if ~isempty(startrh) && isempty(endrh)
        rwdnow(npos) = 1;
    end
    
    % visited both
    if ~isempty(startrh) && ~isempty(endrh)
        if startrh < endrh
            rwdnow(npos) = 1;
        else
            rwdnow(1) = 1;
        end
    end
end


if strcmp(rp,'StartRefbyEnd')
    %  rewards placed at track start, refresh when reach track end
    % this is tough as it requires planning
    behpast = Sh(1:end);
    startrh = find(behpast == 1,1,'first');
    endrh = find(behpast == npos,1,'first');
    
    rwdnow = zeros(1,npos);
    % never visited both ends
    if isempty(startrh) && isempty(endrh)
        rwdnow(1) = 1;
    end
    
    % never visited start but visited end
    if isempty(startrh) && ~isempty(endrh)
        rwdnow(1) = 1;
    end
    
    % visited both
    if ~isempty(startrh) && ~isempty(endrh)
        if startrh < endrh
            % recently reach start, remove reward
            rwdnow = zeros(1,npos);
        else
            % recently reach end, reward at start
            rwdnow(1) = 1;
        end
    end
end


if strcmp(rp,'StartMid')
    % alternative rewards placed at track start or track middle
    midbin = round(npos/2);
    behpast = Sh(1:end);
    startrh = find(behpast == 1,1,'first');
    midrh = find(behpast == midbin,1,'first');
    
    rwdnow = zeros(1,npos);
    % never visited start or mid
    if isempty(startrh) && isempty(midrh)
        rwdnow(1) = 1;
    end
    
    % never visited start but visited mid
    if isempty(startrh) && ~isempty(midrh)
        rwdnow(1) = 1;
    end
    
    % never visited mid but visited start
    if ~isempty(startrh) && isempty(midrh)
        rwdnow(midbin) = 1;
    end
    
    % visited both
    if ~isempty(startrh) && ~isempty(midrh)
        if startrh < midrh
            rwdnow(midbin) = 1;
        else
            rwdnow(1) = 1;
        end      
    end
end

%% for T maze
% for T maze, if we have npos position bins, then we will have floor(npos/3)
% bins for left and right arms, and remaining are center arms
% we will organize bins as center-left-right
if strcmp(rp,'TLeftRight')
    nright = floor(npos/3);
    nleft = nright;
    ncenter = npos - nleft - nright;
    % alternative rewards placed at left and right arms, then reset at
    % middle arm
    behpast = Sh(1:end);

    % initially reward at start arm
    rwdnow = zeros(1,npos);
    rwdnow(1) = 1;
    ccollect = 0; % if recently reward collect is at center
    lcollect = 0; % if recently reward collect is at left
    rcollect = 0; % if recently reward collect is at right
    for istep = length(behpast):-1:1
        if ~ccollect && ~lcollect && ~rcollect && behpast(istep) == 1
           % animal first time collect center reward 
           rwdnow = zeros(1,npos);
           rwdnow(ncenter+nleft) = 1; % put reward at left
           ccollect = 1;
        end
        
        if ~ccollect && ~lcollect && rcollect && behpast(istep) == 1
            % animal recenlty collect right reward
            rwdnow = zeros(1,npos);
            rwdnow(ncenter+nleft) = 1; % put reward at left
            ccollect = 1;
        end
        
        if ~ccollect && lcollect && ~rcollect && behpast(istep) == 1
            % animal recenlty collect left reward
            rwdnow = zeros(1,npos);
            rwdnow(npos) = 1; % put reward at right
            ccollect = 1;
        end
        
        if  rwdnow(ncenter+nleft) == 1 && behpast(istep) == ncenter+nleft
            % animal collect left reward
            rwdnow = zeros(1,npos);
            rwdnow(1) = 1; % put reward back to center
            ccollect = 0;
            lcollect = 1;
            rcollect = 0;
        end
        
        if  rwdnow(npos) == 1 && behpast(istep) == npos
            % animal collect right reward
            rwdnow = zeros(1,npos);
            rwdnow(1) = 1; % put reward back to center
            ccollect = 0;
            lcollect = 0;
            rcollect = 1;
        end
    end
end
end

