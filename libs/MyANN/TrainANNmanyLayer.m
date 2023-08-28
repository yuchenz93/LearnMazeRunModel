function [net,pctloss,ax1,ax2] = TrainANNmanyLayer(net,x,t,lf,Tpara)
%TrainANNmanyLayer this function train a neural network nased based on inputs x,
%targets t, and user defined loss function lf, the ANN use sigmoid function
%at each layer


% inputs  net,  neural network with net.W{} being weight matrix across
%               layers, and net.b{} being constant vector across layers 
%               at each layer, it takes a column vector and produce a
%               column vector (column matrix)
%          x,   inputs, columns being realizations and rows with inputs DOF
%          t,   targets, columns being realizations and rows with targets DOF
%          lf,  user defined loss function, should be determined by x,y and t
%               the input should be net = TrainANNwL(net,x,t,@mylossfunction)
% output   net,  trained neural network

% plot convergence

%% define and derive parameters
nlayer = length(net.W);
% we want to know the parameter size
nsize = nan(1,nlayer+1);
for ilayer = 1:nlayer
    nsize(ilayer) = size(net.W{ilayer},1);
end
nsize(nlayer+1) = size(t,1);
nsample = size(x,2);

itlim = Tpara.RTsteps; % upper limit of iterations
alpha = Tpara.RTalpha; % learning rate, we will use momentum to accelerate
mommu = Tpara.RTMom; %  we will use momentum to accelerate
L2R = Tpara.RTL2; % L2 regularization coefficient
pcvg = Tpara.RTPlot; % plot training loss
npos = length(net.b{end});
tol = 1e-8; % pct covergence criteria of outputs
yori = zeros(nsize(nlayer+1),nsample);

%% learning loop iternation
ax1 = [];
ax2 = [];
if pcvg
    ferr = figure;
    ax1 = subplot(1,2,1);
    hold(ax1,'on')
    set(ax1,'yscale','log')
    xlabel(ax1,'iterations')
    ylabel(ax1,'Total Loss')
    
    ax2 = subplot(1,2,2);
    hold(ax2,'on')
    set(ax2,'yscale','log')
    xlabel(ax2,'iterations')
    ylabel(ax2,'Measured Loss')
end
mdW = cell(1,nlayer);
mdb = cell(1,nlayer);
for it = 1:itlim
    %% first we want to know outputs, get derivative of loss function with outputs
    ynow = x;
    for ilayer = 1:nlayer
        ynow = sigmoid(net.W{ilayer}*ynow + net.b{ilayer});
    end
    
    % check convergence
    ydiff = ynow - yori;
    pctdiff = norm(ydiff,"fro")/norm(ynow,"fro");
    if pctdiff <= tol && it >= 10
        break;
    end
    yori = ynow;
    
    % get derivative of loss function with respect to outputs
    if isempty(Tpara.Dis)
        [alldLdy,pctloss] = feval(lf,x,ynow,t,npos);
    else
        [alldLdy,pctloss] = feval(lf,x,ynow,t,Tpara.Dis);
    end
    
    if Tpara.test
        plotstep = 10;
    else
        plotstep = 100;
    end
    if pcvg && rem(it,plotstep) == 0 % plot every 10 steps
        error = norm(ynow-t,"fro")/norm(t,"fro");
        %        plot(it,pctdiff,'or')
        plot(ax1,it,error,'ob')
        plot(ax2,it,pctloss,'or')
        xlabel('iterations')
        drawnow
    end
    
    % predefine gradiant matrix
    dW = cell(1,nlayer);
    db = cell(1,nlayer);
    for ilayer = 1:nlayer
        dW{ilayer} = zeros(size(net.W{ilayer}));
        db{ilayer} = zeros(size(net.b{ilayer}));
    end
    %% first use forward propagation to get necessary parameters
    allz = cell(1,nlayer);
    alldz = cell(1,nlayer);
    alla = cell(1,nlayer+1);
    alla{1} = x;
    for ilayer = 1:nlayer
        atmp = net.W{ilayer}*alla{ilayer};
        allz{ilayer} = atmp + repmat(net.b{ilayer},[1,nsample]);
        % linear operation in the layer
        alldz{ilayer} = dsigmoid(allz{ilayer});
        % derivative of sigmoid of linear operation result
        
        alla{ilayer+1} = sigmoid(allz{ilayer});
        % alla can be viewed as inputs to each layer, the last one is
        % y, is not inputs to any layer but the last outputs
    end
    %% how loss function is dependent on outputs
    dLdy = alldLdy;
    %% use backpropagation to get gradients
    dLdo = cell(1,nlayer); % derivative of loss function with respect to output of each layer
    dLdz = cell(1,nlayer);
    for ilayer = nlayer:-1:1
        if ilayer == nlayer
            dLdo{ilayer} = dLdy;
            dLdz{ilayer} = alldz{ilayer} .* dLdo{ilayer};
        else
            dLdo{ilayer} = net.W{ilayer+1}' * dLdz{ilayer+1};
            dLdz{ilayer} = alldz{ilayer} .* dLdo{ilayer};
        end
        
        % dLdz is also dLdb as z = wx+b, sum over realizations
        dLdb = sum(dLdz{ilayer},2);
        % add regularization term
        dLdb = dLdb + L2R*2*net.b{ilayer}*nsample;
        db{ilayer} = dLdb;
        
        % for dLdw
        % dL/dw = dL/do * do/dz * dz/dw
        % where dL/do is nout*nsample do/dz is nout*nsample, we should do
        % element wise multiple between them, we get dL/dz nout*nsample
        % then dL/dw = dL/dz * dz/dw where dz/ds is the input to this layer
        % with dimension nsample * nin, in this way, we sum over all the
        % samples
        
        dLdw = (dLdo{ilayer} .* alldz{ilayer}) * alla{ilayer}'; % sums over realizations
        % add regularization term
        dLdw = dLdw + L2R*2*net.W{ilayer}*nsample;
        dW{ilayer} = dLdw;
    end
    %% update matrices
    for ilayer = 1:nlayer
        if it == 1
            mdW{ilayer} = dW{ilayer}/nsample;
            mdb{ilayer} = db{ilayer}/nsample;
        else
            mdW{ilayer} = mommu * mdW{ilayer} + dW{ilayer}/nsample;
            mdb{ilayer} = mommu * mdb{ilayer} + db{ilayer}/nsample;
        end
        
        net.W{ilayer} = net.W{ilayer} - alpha .* mdW{ilayer};
        net.b{ilayer} = net.b{ilayer} - alpha .* mdb{ilayer};
    end
end
if pcvg && ~Tpara.test
   close(ferr)
end
end

