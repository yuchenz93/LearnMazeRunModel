% this script stimulate rats behavior on a familiar linear track but
% unknown reward policy, we want to study how the animal learn to run
% directionaly on the linear track if we given rewards alternatively at
% track ends
% in this version, animal would update reward policy whenever the estimated
% reward is quite different from actual reward 

clear;clc;close all
dbstop if error
%% set parameters
% general parameters
nses = 5; % we have 5 sessions in total
npos = 20; % number of position bins
tlen = 2*npos;  
% memory length, should be larger than npos  so the animal is possible to 
% remember when it reaches rewards
seqlen = 1e3; % toal time steps in each session 
rp = 'altends'; %reward policy, put reward at maze ends
% see ActRewardPolicy.m for options

% Train parameters
Tpara.confidiscount = 0.2; % confidence discount factor
Tpara.RWStep = 8; % Gaussian smooth kernel of random walker, large value leads to more diffusive random walker
Tpara.samplelimit = 1e3; % sample limit of training, we use past ... samples to train
Tpara.RTLossThreshold = 0.6; % retrain reward policy when measured loss excceed this value
Tpara.tgapcri = 25; % train at most once every 25 steps
Tpara.Actsteps = 6; % iteration limits of action policy planning
Tpara.RTsteps = 1e3; % iteration limits of reward policy training
Tpara.RTalpha = 0.3; % reward training learning rate
Tpara.RTMom = 0.95; % momentum factor to accelerate reward training
Tpara.RTL2 = 0.0001; % L2 regularization of reward training, usually smaller than 0.01
% use even smaller values if the task is more difficult
Tpara.RTPlot = 0; % whether or not plot reward training loss
Tpara.ActGamma = 0.95; % reward discount factor in determining action policy
Tpara.TeleP = 40; % coefficient of teleportation panelty function
Tpara.Dis = []; % distance matrix, we don't need this for linear track 

% path to save figures
OutPath = '/media/yuchen/data14/CAResults/DirectionRun/L2R_LbySurprise/';
savepath = [OutPath,rp,filesep];
mkdir(savepath)
Tpara.test = 0;
runtest = 20; % run it several times
%% Pilot study for new parameters
if Tpara.test
   ParaPilotTest(npos,tlen,rp,Tpara) 
   return
end

%% test loop
for iplot = 1:runtest
    %% Train animal and plot the trajectory
    % parameter initaization
    confi = 0; % initial confidence of reward model
    rwdacc = 0; % accumulate reward with temporal discount
    allact = [];  % behavior history
    allrwd = [];  % reward history to learn
    allestrwd = []; % estimated reward history
    
    % we use a step learning algorithm.
    % With hypothesized reward policy, animal compute optimal action policy offline,
    % then animal explore the track with that action policy, and update reward
    % policy based on experience
    
    % the reward policy is a neural network with behavior history as input and
    % reward across bins as output
    % we will first initialize the network
    net = ANNofR_Initialize_v2(tlen,npos);
    figure
    set(gcf,'outerposition',get(0,'screensize'));
    [~,pos]  = tight_subplot(nses,1,[0.08 0.08],[0.08 0.08],[0.05 0.05],0);
    
    % session loops, we just reset the position between sessions, the "memory"
    % of reward history and reward model is inherited across sessions
    for ises = 1:nses
        disp(['In session ',num2str(ises),'...'])
        
        ax = subplot('position',pos{ises});
        yyaxis left
        hold(ax,'on')
        ylim(ax,[1 npos])
        xlim(ax,[1 seqlen])
        xlabel(ax,'TimeSteps')
        ylabel(ax,'Linear position')
        
        yyaxis right
        hold(ax,'on')
        ylim(ax,[0 1])
        ylabel(ax,'Confidence level')
        title(ax,['Session',num2str(ises)])
        set(ax,'TickLength',[0.002, 0.002])
        % based on hypothesized reward policy, animal come up with a action
        % policy and behave based on that, then it will update the reward
        % policy based on experience
        [Act,net,confi,rwdacc,allact,allrwd,allestrwd] = DynShRT_inst_v2(tlen,npos,seqlen,net,rp,confi,rwdacc,allact,allrwd,allestrwd,Tpara,ax);
        if ises == 1
            hd(1) = plot(nan,nan,'-k');
            hd(2) = plot(nan,nan,'or');
            hd(3) = plot(nan,nan,'-','color',[0.56,0.56,0.3]); 
            legend(hd,{'Trajectory','Reward collected','Confidence Level'})
        end
    end
    stt = ['Npos',num2str(npos),' Test',num2str(iplot)];
    File_Path = strcat(savepath,stt,'.fig');
    saveas(gcf, File_Path);
    File_Path = strcat(savepath,stt,'.jpg');
    saveas(gcf, File_Path);
    close all
end