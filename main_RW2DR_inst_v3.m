% this script stimulate rats behavior on a familiar linear track but
% unknown reward policy, we want to study how the animal learn to run
% directionaly on the track 
% in this version, we use distance matrix so we can include T maze, also,
% we use limited past position and limited reward history to predict reward

clear;clc;close all
dbstop if error
%% set parameters
% general parameters
nses = 3; % we have 3 sessions in total
npos = 30; % number of position bins
postlen = 20;  % memory length of position
rwdtlen = 5;  % memory length of reward history
seqlen = 2e3; % toal time steps in each session 
rp = 'TLeftRight'; %reward policy, put reward at maze ends
% see ActRewardPolicy_Dynamic.m for options

% Train parameters
Tpara.confidiscount = 0.2; % confidence discount factor
Tpara.RWStep = 5; % Gaussian smooth kernel of random walker, large value leads to more diffusive random walker
Tpara.samplelimit = 4e3; % sample limit of training, we use past ... samples to train
Tpara.RTLossThreshold = 0.3; % retrain reward policy when measured loss excceed this value
Tpara.tgapcri = 25; % train at most once every 25 steps
Tpara.uptgapcri = 200; % train at least once every 200 steps
Tpara.Actsteps = 6; % iteration limits of action policy planning
Tpara.RTsteps = 2e3; % iteration limits of reward policy training
Tpara.RTalpha = 0.3; % reward training learning rate
Tpara.RTMom = 0.95; % momentum factor to accelerate reward training
Tpara.RTL2 = 0.0001; % L2 regularization of reward training, usually smaller than 0.01
Tpara.BatchSize = 0.1; % each iteration use a small portion to get gradiant (0-1)
% use even smaller values if the task is more difficult
Tpara.RTPlot = 0; % whether or not plot reward training loss
Tpara.ActGamma = 0.95; % reward discount factor in determining action policy
Tpara.TeleP = 40; % coefficient of teleportation panelty function
Tpara.Errrange = 6; % if empty we will use default setting round(npos/5)
Tpara.Dis = AllDisM(npos,rp); % distance matrix

% path to save figures
OutPath = '/media/yuchen/data14/CAResults/DirectionRun/DisM_RWDHistory/';
savepath = [OutPath,rp,filesep];
mkdir(savepath)
Tpara.test = 0;
runtest = 10; % run it several times
%% Pilot study for new parameters
if Tpara.test
   ParaPilotTest_v3(npos,postlen,rwdtlen,rp,Tpara) 
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
    net = ANNofR_Initialize_v2(postlen+rwdtlen,npos);
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
        
        if strcmp(rp,'TLeftRight')
           yyaxis left
           narm = floor(npos/3);
           ncenter = npos-2*narm;
           fill(ax,[1 seqlen seqlen 1],[ncenter+1 ncenter+1 ncenter+narm+0.5 ncenter+narm+0.5],[0.76 0.54 0.44],'LineStyle','none','FaceAlpha',0.4) 
           fill(ax,[1 seqlen seqlen 1],[ncenter+narm+0.5 ncenter+narm+0.5 npos npos],[0.44 0.54 0.76],'LineStyle','none','FaceAlpha',0.4) 
        end
        % based on hypothesized reward policy, animal come up with a action
        % policy and behave based on that, then it will update the reward
        % policy based on experience
        [Act,net,confi,rwdacc,allact,allrwd,allestrwd] = DynShRT_inst_v3(postlen,rwdtlen,npos,seqlen,net,rp,confi,rwdacc,allact,allrwd,allestrwd,Tpara,ax);
        if ises == 1
            if strcmp(rp,'TLeftRight')
                hd(1) = plot(nan,nan,'-k');
                hd(2) = plot(nan,nan,'or');
                hd(3) = plot(nan,nan,'-','color',[0.56,0.86,0.3]);
                hd(4) = plot(nan,nan,'-','linewidth',6,'color',[0.76,0.54,0.44]);
                hd(5) = plot(nan,nan,'-','linewidth',6,'color',[0.44,0.54,0.76]);
                legend(hd,{'Trajectory','Reward collected','Confidence Level','Left Arm','Right Arm'})
            else
                hd(1) = plot(nan,nan,'-k');
                hd(2) = plot(nan,nan,'or');
                hd(3) = plot(nan,nan,'-','color',[0.26,0.56,0.1]);
                legend(hd,{'Trajectory','Reward collected','Confidence Level'})
            end
        end
    end
    stt = ['Npos',num2str(npos),' Test',num2str(iplot)];
    File_Path = strcat(savepath,stt,'.fig');
    saveas(gcf, File_Path);
    File_Path = strcat(savepath,stt,'.jpg');
    saveas(gcf, File_Path);
    close all
end