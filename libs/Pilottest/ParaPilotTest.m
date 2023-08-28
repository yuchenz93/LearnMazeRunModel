function ParaPilotTest(npos,tlen,rp,Tpara)
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
    Tpara.samplelimit = nsample;
    x = nan(nsample,tlen);
    % this is the history of behavior, rows are samples, columns are animals
    % linear position at each time step,
    % from 1 to tlen it means current, 1 step before, 2 steps before ...
    
    y = zeros(nsample,npos);
    % this is the reward policy, rows are samples, columns are animals
    % rewards at each site
    
    % the initial state if always one fixed random state
    s0 = randi(npos);
    Sh0 = ones(1,tlen)*s0;
    xlposall = s0;
    % pre define result
    Shall = cell(1,nsample + 1);
    Shall{1} = Sh0;
    
    for istep = 2:nsample + 1
        rwdnow = ActRewardPolicy(xlposall,npos,rp);
        if isrow(rwdnow)
            rwdnow = rwdnow';
        end
        % based on reward, optimize action policy,
        % specifically, if the estimated reward is similar to previous step, we
        % need not change action policy, one way to realize it is to use
        % previous action policy as the intial state
        gstd = Tpara.RWStep;
        Trdwk = RandomWalkerPolicy(npos,gstd);
        
        % get next position based on tnow
        xnext = NextbyTM(Trdwk,Shall{istep-1}(1));
        nowSh = nan(1,tlen);
        % the first element of next sequence is the next state based on action
        % policy, the remaining elements are the history of the states which
        % can be obatined from previous sequence
        
        nowSh(1) = xnext;
        nowSh(2:end) = Shall{istep-1}(1:end-1);
        Shall{istep} = nowSh;
        
        x(istep-1,:) = nowSh;
        y(istep-1,:) = rwdnow';
    end
    
%     %% try the net that's going to work
%     net = RewardatEndANN(tlen,npos);
%     
%     esty = nan(size(y));
%     for isample = 1:nsample
%         xnow = x(isample,:);
%         xnow = xnow';
%         hd1 = sigmoid(net.W{1}*xnow + net.b{1});
%         hd2 = sigmoid(net.W{2}*hd1 + net.b{2});
%         hd3 = sigmoid(net.W{3}*hd2 + net.b{3});
%         yout = sigmoid(net.W{4}*hd3 + net.b{4});
%         esty(isample,:) = yout';
%     end
%     diff = y-esty;
    
    %% train the whole network
    net2 = ANNofR_Initialize_v2(tlen,npos);
    [net2,losspct,ax1,ax2] = TrainANNmanyLayer(net2,x',y',@VCNTy_TrackRewardLoss,Tpara);
    esty2 = nan(size(y));
    yerr = nan(1,nsample);
    for isample = 1:nsample
        xnow = x(isample,:);
        xnow = xnow';
        hd1 = sigmoid(net2.W{1}*xnow + net2.b{1});
        hd2 = sigmoid(net2.W{2}*hd1 + net2.b{2});
        hd3 = sigmoid(net2.W{3}*hd2 + net2.b{3});
        yout = sigmoid(net2.W{4}*hd3 + net2.b{4});
        esty2(isample,:) = yout';
        yerr = norm(esty2(isample,:) - y(isample,:))./norm(y(isample,:));
    end
    plot(ax1,[oristep oristep],ylim(ax1),'--r')
    plot(ax2,[oristep oristep],ylim(ax1),'--r')
    stt = ['Training with Sample Size ',num2str(nsample)];
    suptitle(stt)
    merr = nanmean(yerr);
    disp(['Mean normalized error:',num2str(merr),' with sample size ',num2str(nsample),' and train iteration ', num2str(Tpara.RTsteps)])
end
end