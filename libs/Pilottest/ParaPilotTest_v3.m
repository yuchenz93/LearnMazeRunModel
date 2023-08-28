function ParaPilotTest_v3(npos,tlen,trwd,rp,Tpara)
% this function runa pilot study to check if the model is likely to work
% with a given parameter space

% inputs  npos number of position bins
%         tlen memory length
%         rp   reward policy
%         Tpara  structure with parameter spaces

%% predefine results
orinsample = Tpara.samplelimit; % number of dofs
allsample = [orinsample,2*orinsample,3*orinsample,4*orinsample];
% initialize the net
oristep = Tpara.RTsteps;
Tpara.RTsteps = 2*oristep;
Tpara.RTPlot = 1;
for nsample = allsample
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
    Shall = cell(1,nsample);
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
    %intialize some values
    ccollect = 0;
    lcollect = 0;
    rcollect = 0;
    rwdnow = zeros(1,npos);
    rwdnow(1) = 1;
    allact = [];
    allrwd = [];
    %% action loop
    for istep = 2:nsample + 1
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
        
        % use random walker to move
        Tnow = RandomWalkerPolicy_withDis(Dis,gstd);
        
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
    end
    
    %% train the whole network
    net2 = ANNofR_Initialize_v2(tlen+trwd,npos);
    [net2,losspct,ax1,ax2] = TrainANNmanyLayer_v2(net2,allact,allrwd,@VCNTy_TrackRewardLoss_Dis,Tpara);
    esty2 = nan(size(allrwd));
    yerr = nan(1,nsample);
    for isample = 1:nsample
        xnow = allact(:,isample);
        hd1 = sigmoid(net2.W{1}*xnow + net2.b{1});
        hd2 = sigmoid(net2.W{2}*hd1 + net2.b{2});
        hd3 = sigmoid(net2.W{3}*hd2 + net2.b{3});
        yout = sigmoid(net2.W{4}*hd3 + net2.b{4});
        esty2(:,isample) = yout;
        yerr = norm(esty2(:,isample) - allrwd(:,isample))./norm(allrwd(:,isample));
    end
    plot(ax1,[oristep oristep],ylim(ax1),'--r')
    plot(ax2,[oristep oristep],ylim(ax1),'--r')
    stt = ['Training with Sample Size ',num2str(nsample)];
    suptitle(stt)
    merr = nanmean(yerr);
    disp(['Mean normalized error:',num2str(merr),' with sample size ',num2str(nsample),' and train iteration ', num2str(Tpara.RTsteps)])
end
end