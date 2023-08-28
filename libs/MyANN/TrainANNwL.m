function net = TrainANNwL(net,x,t,lf)
%TRAINANNWL this function train a neural network nased based on inputs x,
%targets t, and user defined loss function lf, the ANN use sigmoid function
%at each layer
% this might only support ANN with two layers as I am not sure about how
% Matlab organize net structure with more layers...

% inputs  net,  neural network with default Matlab struct, should contain
%               net.IW{} with net.IW{1} being weight matrix projecting from
%               input layer to hidden layer (num hidden * num inputs);
%               net.LW{} with net.LW{2,1} being weight matrix projecting
%               from hidden layer to output layer (num outputs * num hidden)
%               net.b with net.b{1} being constant vector of hidden layer (num hidden * 1);
%               net.b with net.b{2} being constant vector of output layer (num outputs * 1);
%          x,   inputs, columns being realizations and rows with inputs DOF
%          t,   targets, columns being realizations and rows with targets DOF
%          lf,  user defined loss function, should be determined by x,y and t
%               the input should be net = TrainANNwL(net,x,t,@mylossfunction)
% output   net,  trained neural network

% we have an optimized version net = TrainANNwLmanyLayer(net,x,t,lf) which
% is faster and support more layers

%% define and derive parameters
% for convenience, we will use
% n to represent number of inputs,
% m to represents number of hidden elements,
% p to represents number of outputs,
% q to represents realizations
% thus we have the dimensions:
% x(n*q)  t(p*q)  hd(m*q) net.IW{1}(m*n)  net.LW{2,1}(p*m)  net.b{1}(m*1)  net.b{2}(p*1)
n = size(x,1);
m = size(net.IW{1},1);
p = size(net.LW{2,1},1);
q = size(x,2);

itlim = 20; % upper limit of iterations
alpha = 0.2; % learning rate
tol = 1e-4; % pct covergence criteria of outputs
yori = zeros(p,q);
% plot convergence
pcvg = 1;
%% learning loop iternation
if pcvg
    ferr = figure;
    hold on
end
for it = 1:itlim
    %% first we want to know outputs, get derivative of loss function with outputs
    % hidden layer before sigmoid, we also call it hdz
    hdz = net.IW{1}*x + net.b{1}; %m*q
    % after sigmoid we get hidden layer
    hd = 1./(1+exp(-hdz)); % m*q
    
    % output layer before sigmoid, we also call it outz
    outz = net.LW{2,1}*hd + net.b{2}; % p*q
    % after sigmoid we get output layer
    y = 1./(1+exp(-outz)); % p*q
    
    % check convergence
    ydiff = y - yori;
    pctdiff = norm(ydiff,"fro")/norm(y,"fro");
    if pctdiff <= tol
        break;
    end
    yori = y;
    
    if pcvg
       error = norm(y-t,"fro")/norm(t,"fro");
       plot(it,pctdiff,'or')
       plot(it,error,'ob')
    end
    % get derivative of loss function with respect to outputs
    alldLdy = feval(lf,x,y,t);
    
    dLdb2sum = zeros(p,1); % gradient of outlayer b   net.b{2}
    dLdw2sum = zeros(p,m); % gradient of outlayer w   net.LW{2,1}
    dLdb1sum = zeros(m,1); % gradient of hiddenlayer b   net.b{1}
    dLdw1sum = zeros(m,n); % gradient of hiddenlayer w   net.IW{1}
    %% we need to do it for each realization... and we will sum over realizations
    for isample = 1:q
        xnow = x(:,isample); % n*1
        %% first use forward propagation to get necessary parameters
        % get hidden layer before sigmoid,  we also call it hdz,
        % rows are numhidden, columns are realizations
        hdz = net.IW{1}*xnow + net.b{1}; %m*1
        % get derivative of sigmoid on hidden layer
        hdzdsig = dsigmoid(hdz); % m*1
        hd = 1./(1+exp(-hdz)); % m*1
        
        % output layer before sigmoid, we also call it outz
        outz = net.LW{2,1}*hd + net.b{2}; % p*1
        % get derivative of sigmoid on output layer
        outzdsig = dsigmoid(outz); % p*1
        % rows are num output, columns are realizations
        
        %% how loss function is dependent on outputs
        dLdy = alldLdy(:,isample); % we expect it to be p*1, dependents on each output
        
        %% use backpropagation to get gradients
        % note loss function as L,note z = wh+b,z is the input to the sigmoid
        % gradients of z of output layer
        % dL/dz = dL/dy * dy/dz =  dL/dy * dsigmoid(z)
        dLdoutz = outzdsig .* dLdy; % p*1
        
        % gradients of constant vector of output layer, should have same size
        % with net.b{2} (p*1)
        % dL/db2 = dL/dy*dy/db2  dy/db2 = d(sigmoil(w2*h+b2))/db2 = dsigmoid(w2*h+b2)
        dydb2 = outzdsig; % p*1
        dLdb2 = dydb2 .* dLdy; % p*1
        dLdb2sum = dLdb2sum + dLdb2;
        
        % gradients of weight matrix of output layer, should have same size
        % with net.LW{2,1} (p*m)
        % dL/dw2 = dL/dy*dy/dw2  dy/dw2 = d(sigmoil(w2*h+b2))/dw2 = h*dsigmoid(w2*h+b2)
        dydw2 = outzdsig * hd' ; % (p*1) * (1*m) = p*m
        dLdw2 = zeros(p,m);
        for iout = 1:p
            dLdw2(iout,:) = dydw2(iout,:) * dLdy(iout); % p*m same size with net.LW{2,1}
        end
        dLdw2sum = dLdw2sum + dLdw2;
        
        
        % gradients of constant vector of hidden layer, should have same size
        % with net.b{1} (m*1)
        % dL/db1 = dL/dhda * dhda/dhdz * dhdz/db1 = dL/dhda * hdzdsig
        % dL/dhda = wout*dL/dLdoutz
        dLdhda = net.LW{2,1}' * dLdoutz; % (m*p) * (p*1) = (m*1)
        % we expect this to be m*1
        dLdb1 = dLdhda.* hdzdsig;
        dLdb1sum = dLdb1sum + dLdb1;
        
        % gradients of weight matrix of hidden layer, should have same size
        % with net.IW{1} (m*n)
        % dL/dw1 = dL/dhda * dhda/dhdz * dhdz/dw1 = dL/dhda * hdzdsig * x
        % dL/dhda = wout*dL/dLdoutz
        dLdhdax = dLdhda * xnow'; %(m*1) * (1*n) = m*n
        dLdw1 = zeros(m,n);
        for ihd = 1:m
            dLdw1(ihd,:) = dLdhdax(ihd,:) * hdzdsig(ihd);
        end
        dLdw1sum = dLdw1sum + dLdw1;
    end
    %% update matrices
    net.LW{2,1} = net.LW{2,1} - alpha .* dLdw2sum/q;
    net.IW{1} = net.IW{1} - alpha .* dLdw1sum/q;
    net.b{2} = net.b{2} - alpha .* dLdb2sum/q;
    net.b{1} = net.b{1} - alpha .* dLdb1sum/q;
end

close(ferr)
end

