function Dis = TMazeDisMatrix(npos)
% TMazeDisMatrix returns the distance matrix for a T maze
% we will organize postion bin index as center-left-right
% left and right arms have floor(npos/3) bins
% inputs: npos     total number of position bins
% output: Dis      Distance matrix

nright = floor(npos/3);
nleft = nright;
ncenter = npos - nleft - nright;
    
Dis = zeros(npos,npos);

% we can view center+left arm as a linear track, the distance is just the
% difference in index
for ir = 1:ncenter+nleft
    for jr = 1:ncenter+nleft
        Dis(ir,jr) = abs(ir-jr);
    end
end
% construct right arm
for ir = ncenter+nleft+1:npos
    % distance between right and right
    for jr = ncenter+nleft+1:npos
        Dis(ir,jr) = abs(ir-jr);
    end
    % distance between right and center
    for jr = 1:ncenter
        Dis(ir,jr) = ir - (ncenter+nleft) + ncenter-jr;
        Dis(jr,ir) = ir - (ncenter+nleft) + ncenter-jr;
    end
    % distance between right and left
    for jr = ncenter+1:ncenter+nleft
        Dis(ir,jr) = ir - (ncenter+nleft) + jr-ncenter;
        Dis(jr,ir) = ir - (ncenter+nleft) + jr-ncenter;
    end
end

end

