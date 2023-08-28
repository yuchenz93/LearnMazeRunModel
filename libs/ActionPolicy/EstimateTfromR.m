function Tall = EstimateTfromR(r,varargin)
% EstimateTfromR  optimize the action policy based on reward policy, the
% action policy is represented as a state transition matrix
% inputs   r,   reward column vecotr indicates the reward at each site
% optional inputs:
%         itlim,  upper limit of iteration steps [5]
%         gamma,  discount factor in time [0.95]
%         calpha, coefficient before teleportation penatly function [0.1]
%         epi,  step size used to compute gradient [0.05]
%         learnrate,  learning rate of gradient descend [0.2]
%         threshold,  threshold of convergence [1e-5]
%         Tinitial,  initial transition matrix [random]
%         confi,   the confidence level of reward estimation [0.5]
%                  if not confident, apply random walker
% output:  optimized transition matrix with limited optimization steps
%% processing inputs
% make sure the reward is a column vector
if isrow(r)
    r = r';
end

npos = length(r);

args.itlim = 5; 
% optmizing steps, we don't want this to be large as we want it can make
% quick decisions
args.gamma = 0.95; % discount factor
args.calpha = 40; % factor of the panelty function against teleportation
args.epi = 0.05; % step size to compute gradient
args.learnrate = 0.2; % learning rate
args.threshold = 1e-5; % convergence threshold
args.Tinitial = rand(npos,npos); % initial transition matrix
args.confi = 0.5; 
args.RWk = 8;
args.RWT = [];
args.Dis = [];
args = parseArgs(varargin, args);

gamma = args.gamma;
calpha = args.calpha;
itlim = args.itlim; 
epi = args.epi;
learnrate = args.learnrate;
threshold = args.threshold;
Tinitial = args.Tinitial;

%% construct reward function and cost function
% now we want to use sigmoid function to allow low speed running but
% stronger inhibit fast run
if isempty(args.Dis)
    c = zeros(npos,npos);
    for ir = 1:npos
        for ic = 1:npos
            dis = abs(ir-ic);
            sigdis = 1/(1+exp(-(dis/1.5-6)));
            c(ir,ic) = calpha * sigdis;
        end
    end
else
    % if the distance matrix is give
    dis = args.Dis;
    sigdis = 1./(1+exp(-(dis./1.5-6)));
    c = calpha .* sigdis;

end
%% initilize transition function
T = Tinitial;
% normalize the row
for ir = 1:npos
    T(ir,:) = T(ir,:)./sum(T(ir,:));
end

% we assume a uniform intial position
b0 = ones(1,npos)/npos;

%% optimization loop
vold = nan;

for istep = 1:itlim
   % get expected cost to leave
   el = diag(T*c');
   rnew = r-el;

   % get value function
   disT = eye(npos)-gamma*T;
   v = b0/disT*rnew;  
%    eler = inv(disT);
%    v = b0*eler*rnew; 

   % check convergency
   if ~isnan(vold)
       resi = abs(v-vold)/abs(vold);
       if resi < threshold
          break 
       end
   end
   
   % we want to get gradient by pertubate each element a little bit
   vpb = nan(npos,npos);
   for ir = 1:npos
       for ic = 1:npos
           Ttmp = T;
           Ttmp(ir,ic) = T(ir,ic) + epi;
           for jr = 1:npos
               Ttmp(jr,:) = Ttmp(jr,:)./sum(Ttmp(jr,:));
           end
           el2 = diag(Ttmp*c');
           rnew2 = r-el2;
           disT2 = eye(npos)-gamma*Ttmp;
           vpb(ir,ic) = b0/disT2*rnew2;
%            eler2 = inv(eye(npos)-gamma*Ttmp);
%            vpb(ir,ic) = b0*eler2*rnew2;
       end
   end
    
   % get the gradient
   dpb = vpb-v;
   dpb = dpb./epi;

   % update the transition matrix 
   T = T + dpb*learnrate;
   % make sure the minimum value should be 0
   T(T<0) = 0;
   % normalize the row
   for ir = 1:npos
       T(ir,:) = T(ir,:)./sum(T(ir,:));
   end
   vold = v;
end

%% mix the optimal T from reward estimation with a random walker model based on confidence level
if isempty(args.RWT)
    gstd = args.RWk;
    Trdwk = RandomWalkerPolicy(npos,gstd);
else
    % if random walker transition matrix is defined
    Trdwk = args.RWT;
end
% suppress stationary to encourage exploration
for ir = 1:npos
    Trdwk(ir,ir) = Trdwk(ir,ir)*0.1;
end
for ir = 1:npos
    T(ir,:) = T(ir,:)./sum(T(ir,:));
end

Tall =  Trdwk*(1.1-args.confi) + T*args.confi;
% normalize the row
for ir = 1:npos
    Tall(ir,:) = Tall(ir,:)./sum(Tall(ir,:));
end
end