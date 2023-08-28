function Shall = DynShRT(tlen,npos,seqlen,net)
% this function generate a sequence based on the dynamic estimate of reward
% and corresponding transition matrix
% inputs: tlen, memory length of states, that will determine the estimated
%               reward site
%         npos, total number of position bins
%         seqlen,  sequence length
%         net,  a neural network that predict reward based on behavior
%         history
% outputs: Shall,  generated sequence as a cell array


% the initial state if always one fixed random state
s0 = randi(npos);
Sh0 = ones(1,tlen)*s0;

% pre define result
Shall = cell(1,seqlen);
Shall{1} = Sh0;

% we will dynamically generate new sequence
% we first use a random walker policy as intial transition matrix, this is
% the initial transition matrix of optimization
gstd = 5;
Tnow = RandomWalkerPolicy(npos,gstd);


for istep = 2:seqlen
    % based on current state history, estimate reward
    rnow = net(Shall{istep-1}');
    % based on reward, optimize action policy, 
    % specifically, if the estimated reward is similar to previous step, we
    % need not change action policy, one way to realize it is to use
    % previous action policy as the intial state
    Tnow = EstimateTfromR(rnow,'Tinitial',Tnow,'itlim',5);
    % get next position based on tnow
    xnext = NextbyTM(Tnow,Shall{istep-1}(1));
    nowSh = nan(1,tlen);
    % the first element of next sequence is the next state based on action
    % policy, the remaining elements are the history of the states which
    % can be obatined from previous sequence

    nowSh(1) = xnext;
    nowSh(2:end) = Shall{istep-1}(1:end-1);
    Shall{istep} = nowSh;
end

end