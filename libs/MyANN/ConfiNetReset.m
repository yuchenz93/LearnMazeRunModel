function net = ConfiNetReset(net,confi)
% this function reset the net if the confidence level is low
% inputs:  net     trained network
%          confi   confidence level 0-1




netrandom = net;
for ilayer = 1:length(net.W)
   numin = size(netrandom.W{ilayer},2);
   numout = size(netrandom.W{ilayer},1);
   factor = 2*sqrt(6/(numin+numout));
   netrandom.W{ilayer} = (rand(size(netrandom.W{ilayer}))-0.5)*factor; 
   netrandom.b{ilayer} = (rand(size(netrandom.b{ilayer}))-0.5)*factor; 
   
   net.W{ilayer} = confi*net.W{ilayer} + (1-confi)*netrandom.W{ilayer};
   net.b{ilayer} = confi*net.b{ilayer} + (1-confi)*netrandom.b{ilayer};
end


end
