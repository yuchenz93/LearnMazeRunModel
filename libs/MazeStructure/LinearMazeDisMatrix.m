function Dis = LinearMazeDisMatrix(npos)
% LinearMazeDisMatrix returns the distance matrix for a linear maze

% inputs: npos     total number of position bins
% output: Dis      Distance matrix



Dis = zeros(npos,npos);


for ir = 1:npos
    for jr = 1:npos
        Dis(ir,jr) = abs(ir-jr);
    end
end


end

