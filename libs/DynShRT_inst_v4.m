function [Shall,net,confi,rwdacc,allact,allrwd,allestrwd] = DynShRT_inst_v4(tlen,trwd,npos,seqlen,net,rp,confi,rwdacc,allact,allrwd,allestrwd,Tpara,ax)
% this function generate a sequence based on the dynamic estimate of reward
% and corresponding transition matrix, also it will update the ANN with
% more experience, it will retrain reward policy whenever the estimated
% reward is quite different from actual reward 
% in the v4 version, animal will use actual local reward to replace
% hypothesized reward
% inputs: tlen, memory length of states, that will determine the estimated
%               reward site
%         npos, total number of position bins
%         seqlen,  sequence length
%         net,  a neural network that predict reward based on behavior
%         history
%         rp,   actual reward policy
%         confi, confidence of the reward model, accumulated from
%         experience
%         Tpara, training parameters
% outputs: Shall,  generated sequence as a cell array
%          net,  updated neural network
%          confi, confidence of the reward model, accumulated from
%          experience

if nargin < 13
   ax = [];
end

%% set parameters

sizelim = Tpara.samplelimit;
% get the distance matrix for this problem
Dis = Tpara.Dis;
%% initialization
% the initial state if always one fixed random state
s0 = randi([2,round(npos/2)]);
xlposall = s0; 
Sh0pos = ones(1,tlen)*s0;
Sh0rwd = zeros(1,trwd);
% we put animal closer to start arm so they are more likely to get reward


% this record the actual trajectory and reward history, the current reward
% is determined by this
% pre define result
Shall = cell(1,seqlen);
Shall{1} = cat(2,Sh0pos,Sh0rwd);

% this records spatial bins when animal got rewards, initially they are all
% 0s indicates no reward history, then we will include bins where animal
% collects rewards
rwdhistory = zeros(1,trwd);

% we will dynamically generate new sequence
% we first use a random walker policy as intial transition matrix, this is
% the initial transition matrix of optimization
gstd = Tpara.RWStep;
RWT = RandomWalkerPolicy_withDis(Dis,gstd);
Tnow = RWT;
%intialize some values
ccollect = 0;
lcollect = 0;
rcollect = 0;
rwdnow = zeros(1,npos);
rwdnow(1) = 1;
%% action loop
tgapcri = Tpara.tgapcri; % we don't need to train too frequently, train at most once in 20 steps
tgap = Tpara.tgapcri; 
for istep = 2:seqlen
    confiori = confi;
    % get actual reward based on reward policy
    % !!! the issue of this step is animal only know actual reward near him, not the
    %     rwdnow across all the spatial bins...
    [rwdnow,ccollect,lcollect,rcollect,ridx] = ActRewardPolicy_Dynamic(rwdnow,xlposall(1),npos,rp,ccollect,lcollect,rcollect);
    if isrow(rwdnow)
        rwdnow = rwdnow';
    end
    if ridx
        % if new reward collected in the previous step, update the reward history
        rwdhistory = cat(2,xlposall(1),rwdhistory);
    end
    
    % based on state history, estimate reward
    estrnow = MyANNEstmanyLayer(net,Shall{istep-1}');
    
    % based on Tpara.Errrange, animal will use actual reward in errrange to
    % replace estimated reward
    xprevious = xlposall(1);
    % find bins close to previous location
    closebin = Dis(xprevious,:) <= Tpara.Errrange;
    rwdact = estrnow;
    rwdact(closebin) = rwdnow(closebin);
    % based on reward, optimize action policy, 
    % specifically, if the estimated reward is similar to previous step, we
    % need not change action policy, one way to realize it is to use
    % previous action policy as the intial state
    Tnow = EstimateTfromR(rwdact,'Tinitial',Tnow,'itlim',Tpara.Actsteps,...
        'confi',confi,'gamma',Tpara.ActGamma,'calpha',Tpara.TeleP,'RWT',RWT,'Dis',Dis);
    
    % get next position based on tnow
    xnext = NextbyTM(Tnow,Shall{istep-1}(1));
    nowSh = nan(1,tlen+trwd);
    xlposall = cat(1,xnext,xlposall);
    % the first element of next sequence is the next state based on action
    % policy, the remaining elements are the history of the states which
    % can be obatined from previous sequence

    nowSh(1) = xnext;
    nowSh(2:tlen) = Shall{istep-1}(1:tlen-1); 
    % in nowSh 1:tlen we record position, tlen+1:end we record reward
    nowSh(tlen+1:end) = rwdhistory(1,1:trwd);
    Shall{istep} = nowSh;
  
    % accumulating new experience to learn
    allact = cat(2,allact,Shall{istep-1}');
    % we use Shall{istep-1} here because actual and estimated reward are
    % all obtained based on previous state
    allrwd = cat(2,allrwd,rwdnow);
    allestrwd = cat(2,allestrwd,estrnow);
    % make sure the sample size is not too large, if reach limit, we
    % only learn the newest experience
    if size(allact,2) > sizelim
        allact = allact(:,end-sizelim:end);
        allrwd = allrwd(:,end-sizelim:end);
        allestrwd = allestrwd(:,end-sizelim:end);
    end
    
    % compare the difference at current step between estimated and actual
    % reward
    estrwd = MyANNEstmanyLayer(net,allact);
    [~,pctloss] = VCNTy_TrackRewardLoss_Dis(allact,estrwd,allrwd,Dis);
%     confi = confi*Tpara.confidiscount + (1-pctloss);
%     % this value should be bounded from 0 to 1
%     confi = max(0,confi);
%     confi = min(1,confi);
    
    if rwdnow(nowSh(1)) == 1
        rwcount = 1;
    else
        rwcount = 0;
    end
    rwdacc = rwdacc*0.95 +rwcount;
    tgap = tgap + 1;
    if (pctloss >= Tpara.RTLossThreshold && rwdacc > 0 && tgap >= tgapcri) || tgap >= Tpara.uptgapcri
        % relearn when there is huge error
%         [net,~] = train(net,learnact,learnrwd);
        % based on the confidence level, we may need to reset the net
        % as it might stuck at local minimum
        net = ConfiNetReset(net,confi);
        for subit = 1:5
            [net,losspct] = TrainANNmanyLayer_v2(net,allact,allrwd,@VCNTy_TrackRewardLoss_Dis,Tpara);
            % we need to train several times in case it doesn't reach low
            % loss function due to limited iterations
            if losspct <= Tpara.RTLossThreshold
                break
            end
        end
        confi = confi*Tpara.confidiscount + (1-losspct);
        % this value should be bounded from 0 to 1
        confi = max(0,confi);
        confi = min(1,confi);
        tgap = 1;
    end
   
    
    if rwdacc <= 0.05 && confi >=0.90
        confi = 0.1;
    % use this to avoid cased animal believe there is no reward, and
    % stop moving, and actually receive no reward, thus confidence is
    % high
    end

    
    if nargin >= 13
       yyaxis left
       if strcmp(rp,'TLeftRight')
           % plot line when they are in the same track, otherwise plot dot
           narm = floor(npos/3);
           ncenter = npos - 2*narm;
           armbinedg = [0.5,ncenter+0.5,ncenter+narm+0.5,npos];
           bini = discretize(nowSh(2),armbinedg);
           binj = discretize(nowSh(1),armbinedg);
           if max(bini,binj) == 3 && bini~=binj
               plot(ax,[istep-1,istep],[nowSh(2),nowSh(1)],'.k')
           else
               plot(ax,[istep-1,istep],[nowSh(2),nowSh(1)],'-k')
           end
       else
           plot(ax,[istep-1,istep],[nowSh(2),nowSh(1)],'-k')
       end
       % mark when animal get reward
       if rwcount
           plot(ax,istep,nowSh(1),'or')
       end
       
       yyaxis right
       plot(ax,[istep-1,istep],[confiori,confi],'-','color',[0.26,0.56,0.1]) 
       drawnow
    end
end

end