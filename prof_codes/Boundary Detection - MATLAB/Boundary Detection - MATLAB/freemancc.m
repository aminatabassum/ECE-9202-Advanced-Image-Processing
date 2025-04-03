function CC = freemancc(B)
%FREEMANCC Compute Freeman chain code, first difference of the chain code
%and shape number of a boundary or contour.
%   CC = FREEMANCC(B) computes the 8-connected Freeman code, the first
%   difference of the chain code and also the shape number for a given
%   boundary or contour B. B must be a P-by-2 matrix, where P is the number
%   of 8-connected boundary pixels for the corresponding region. Each row
%   of B represents the row and column coordinates for a pixel. CC is a
%   structure with four fields:
%
%       StartIdx    Row and column coordinates for the starting pixel of
%                   the boundary B.
%
%       ChainCode   The 8-connected chain code of B. If B is a P-by-2 
%                   matrix of pixels representing a closed contour, then
%                   size of this array is 1-by-(P-1). If B is an open
%                   contour, then size of the chain code is 1-by-P.
%
%       FirstDiff   The first difference of the chain code, representing 
%                   its rotation-invariant equivalent form and having the 
%                   same dimensions as the chain code.
%
%       ShapeNum    The shape number of the boundary, which is the first
%                   difference of the smallest magnitude.
%
%   The transition from current pixel coordinates to the next is computed
%   from the connectivity diagram and tables shown below
%
%
%               3  2  1
%                \ | /
%             4 -- C -- 0
%                / | \
%               5  6  7
%   
%
%           -----------------------------------
%           | row_diff | col_diff | direction |
%           |---------------------------------|
%           |    +1    |     0    |     0     |
%           |    +1    |    +1    |     1     |
%           |     0    |    +1    |     2     |
%           |    -1    |    +1    |     3     |
%           |    -1    |     0    |     4     |
%           |    -1    |    -1    |     5     |
%           |     0    |    -1    |     6     |
%           |    +1    |    -1    |     7     |
%           -----------------------------------
%
%
%   Author: Chaya Narayan
%   Dated: October 2016


% Check inputs
if nargin
    if size(B,2) == 2 % Check for P x 2 input matrix
        if B(1,:) ~= B(end,:) % Check for open contour
            % Computing pixelwise difference between boundary indices
            diffB = diff([B; B(1,:)]);
        else
            diffB = diff(B);
        end
    else
        error('Input array dimension mismatch');
    end
else
    error('Too many input arguments');
end


CC.StartIdx = B(1,:);

% Setting up a mapping mechanism between pixelwise differences and
% 8-connectivity directions
% Check for: (4*row_diff + col_diff + 6) yielding unique values for unique
% directions and mapping these to required values. 

idx([1 2 3 5 7 9 10 11]) = [5 4 3 6 2 7 0 1];
diffB_map = 4*diffB(:,1) + diffB(:,2) + 6;

CC.ChainCode = idx(diffB_map); % Indexing into the direction map

% Computing the first difference
CC.FirstDiff = mod(diff([CC.ChainCode(end), CC.ChainCode])+8,8);

% Create a circular shifted matrix of first differences and then sort rows
SNum_all = sortrows(toeplitz([CC.FirstDiff(1) fliplr(CC.FirstDiff(2:end))], CC.FirstDiff));

% Assign smallest magnitude first difference to shape number
CC.ShapeNum = SNum_all(1,:);

clear SNum_all diffB_map idx diffB;

end