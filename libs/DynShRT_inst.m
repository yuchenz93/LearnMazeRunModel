function [Shall,net,confi,rwdacc,allact,allrwd] = DynShRT_inst(tlen,npos,seqlen,net,rp,confi,rwdacc,allact,allrwd,Tpara,ax)
% this function generate a sequence based on the dynamic estimate of reward
% and corresponding transition matrix, also it will update the ANN with
% more experience
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

if nargin < 11
   ax = [];
end

%% set parameters

sizelim = Tpara.samplelimit;
learnstep = Tpara.learnstep;

%% initialization
% the initial state if always one fixed random state

s0 = randi([2,round(npos/2)]); 
% we put animal closer to start arm so they are more likely to get reward
Sh0 = ones(1,tlen)*s0;
xlposall = s0; 
% this record the actual trajectory, we will determine actual reward based on this
% pre define result
Shall = cell(1,seqlen);
Shall{1} = Sh0;

% we will dynamically generate new sequence
% we first use a random walker policy as intial transition matrix, this is
% the initial transition matrix of optimization
gstd = Tpara.RWStep;
Tnow = RandomWalkerPolicy(npos,gstd);

%% action loop

for istep = 2:seqlen
    % get actual reward based on reward policy
    % !!! the issue of this step is animal only know actual reward near him, not the
    %     rwdnow across all the spatial bins...
    rwdnow = ActRewardPolicy(xlposall,npos,rp);
    if isrow(rwdnow)
        rwdnow = rwdnow';
    end
    
    % based on state history, estimate reward
    rnow = MyANNEstmanyLayer(net,Shall{istep-1}');
    
    % based on reward, optimize action policy, 
    % specifically, if the estimated reward is similar to previous step, we
    % need not change action policy, one way to realize it is to use
    % previous action policy as the intial state
    Tnow = EstimateTfromR(rnow,'Tinitial',Tnow,'itlim',Tpara.Actsteps,...
        'confi',confi,'gamma',Tpara.ActGamma,'calpha',Tpara.TeleP,'RWk',Tpara.RWStep);
    
    % get next position based on tnow
    xnext = NextbyTM(Tnow,Shall{istep-1}(1));
    nowSh = nan(1,tlen);
    xlposall = cat(1,xnext,xlposall);
    
    % the first element of next sequence is the next state based on action
    % policy, the remaining elements are the history of the states which
    % can be obatined from previous sequence

    nowSh(1) = xnext;
    nowSh(2:end) = Shall{istep-1}(1:end-1);
    Shall{istep} = nowSh;
    

    
    % accumulating new experience to learn
    allact = cat(2,allact,nowSh');
    allrwd = cat(2,allrwd,rwdnow);
    
    if rem(istep,learnstep) == 0
        % every 10 steps, relearn
        % make sure the sample size is not too large, if reach limit, we
        % only learn the newest experience
        if size(allact,2) > sizelim
            allact = allact(:,end-sizelim:end);
            allrwd = allrwd(:,end-sizelim:end);
        end
%         [net,~] = train(net,learnact,learnrwd);
        % based on the confidence level, we may need to reset the net
        % as it might stuck at local minimum
        net = ConfiNetReset(net,confi);
        [net,losspct] = TrainANNmanyLayer(net,allact,allrwd,@VCNTy_TrackRewardLoss,Tpara);
        confi = confi*Tpara.confidiscount + (1-losspct);
        % this value should be bounded from 0 to 1
        confi = max(0,confi);
        confi = min(1,confi);
   
    end
    if rwdnow(nowSh(1)) == 1
        rwcount = 1;
    else
        rwcount = 0;
    end
    rwdacc = rwdacc*0.95 +rwcount;
    if rwdacc <= 0.05 && confi >=0.95
        confi = 0.1;
    % use this to avoid cased animal believe there is no reward, and
    % stop moving, and actually receive no reward, thus confidence is
    % high
    end

    
    if nargin >= 11
       plot(ax,[istep-1,istep],[nowSh(2),nowSh(1)],'-k') 
       % mark when animal get reward
       if rwcount
           plot(ax,istep,nowSh(1),'or')
       end
       drawnow
    end
end

end