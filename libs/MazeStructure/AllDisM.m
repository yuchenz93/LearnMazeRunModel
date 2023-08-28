function Dis = AllDisM(npos,rp)
% npos returns the distance matrix given total number of position bins and
% reward policy 

% inputs: npos     total number of position bins
%         rp       reward policy
% output: Dis      Distance matrix


%% for linear maze
if strcmp(rp,'altends') || strcmp(rp,'StartRefbyEnd') || strcmp(rp,'StartMid')
    Dis = LinearMazeDisMatrix(npos);
end


%% for T maze
if strcmp(rp,'TLeftRight') 
    Dis = TMazeDisMatrix(npos);
end


end

