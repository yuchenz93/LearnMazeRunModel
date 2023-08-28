% this script stimulate rats behavior on a familiar linear track but
% unknown reward policy, we want to study how the animal learn to run
% directionaly on the linear track if we given rewards alternatively at
% track ends
%% set parameters
nses = 20; % total train sessions, within session we have the same 
ntrial = 1; % 5 trails in each session
npos = 15; % number of position bins
tlen = 60; 
% memory length, should larger than npos so the animal is possible to 
% remember it reaches rewards
seqlen = 60; % in each trail, run 60 time steps
rp = 'altends'; %reward policy, put reward at maze ends
%% training session loop
% we use a step learning algorithm. 
% With hypothesized reward policy, animal compute optimal action policy offline,
% then animal explore the track with that action policy, and update reward
% policy based on experience

% the reward policy is a neural network with behavior history as input and
% reward across bins as output
% we will first initialize the network
net = ANNofR_Initialize(tlen,npos);
figure
set(gcf,'outerposition',get(0,'screensize'));

for ises = 1:nses
    disp(['In session ',num2str(ises),'...'])
    % based on hypothesized reward policy, animal come up with a action
    % policy and behave based on that
    % we will have several trails
    Act = cell(1,ntrial);
    for it = 1:ntrial
        disp(['   trial ',num2str(it)])
        Act{it} = DynShRT(tlen,npos,seqlen,net);
    end
    
    % based on the experience, animal obtain a training set with reward
    [TAct,TR] = GetTrainingSetfromExp(Act,npos,rp);
    
    % after the session, animal update the hypothesized reward policy based
    % on experience
    net = UpdateRfromExp(TAct,TR,net);
    % we will plot the trajectory of animal from the first trail from every
    % session
    traj = GetTrialTrajectory(Act{1});
    subplot(4,ceil(nses)/4,ises)
    plot(1:length(traj),traj,'-ok')
    ylim([1 npos])
    drawnow
end

