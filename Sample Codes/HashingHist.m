function [f BlkIdx] = HashingHist(PCANet,ImgIdx,OutImg)

addpath('./Utils')


NumImg = max(ImgIdx);
f = cell(NumImg,1);
map_weights = 2.^((PCANet.NumFilters(end)-1):-1:0); % weights for binary to decimal conversion

for Idx = 1:NumImg
  
    Idx_span = find(ImgIdx == Idx);
    NumOs = length(Idx_span)/PCANet.NumFilters(end); % the number of "O"s
    Bhist = cell(NumOs,1);
    
    for i = 1:NumOs 
        
        T = 0;
        ImgSize = size(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1) + 1)});
        for j = 1:PCANet.NumFilters(end)
            T = T + map_weights(j)*Heaviside(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)}); 
            % weighted combination; hashing codes to decimal number conversion
            
            OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)} = [];
        end
        
        
        if isempty(PCANet.HistBlockSize)
            NumBlk = ceil((PCANet.ImgBlkRatio - 1)./PCANet.BlkOverLapRatio) + 1;
            HistBlockSize = ceil(size(T)./PCANet.ImgBlkRatio);
            OverLapinPixel = ceil((size(T) - HistBlockSize)./(NumBlk - 1));
            NImgSize = (NumBlk-1).*OverLapinPixel + HistBlockSize;
            Tmp = zeros(NImgSize);
            Tmp(1:size(T,1), 1:size(T,2)) = T;
            Bhist{i} = sparse(histc(im2col_general(Tmp,HistBlockSize,...
            OverLapinPixel),(0:2^PCANet.NumFilters(end)-1)')); 
        else 
            
            stride = round((1-PCANet.BlkOverLapRatio)*PCANet.HistBlockSize); 
            blkwise_fea = sparse(histc(im2col_general(T,PCANet.HistBlockSize,...
              stride),(0:2^PCANet.NumFilters(end)-1)')); 
            % calculate histogram for each local block in "T"
            
           if ~isempty(PCANet.Pyramid)
                x_start = ceil(PCANet.HistBlockSize(2)/2);
                y_start = ceil(PCANet.HistBlockSize(1)/2);
                x_end = floor(ImgSize(2) - PCANet.HistBlockSize(2)/2);
                y_end = floor(ImgSize(1) - PCANet.HistBlockSize(1)/2);
                
                sam_coordinate = [...
                    kron(x_start:stride:x_end,ones(1,length(y_start:stride: y_end))); 
                    kron(ones(1,length(x_start:stride:x_end)),y_start:stride: y_end)];               
                
                blkwise_fea = spp(blkwise_fea, sam_coordinate, ImgSize, PCANet.Pyramid)';
                
           else
                blkwise_fea = bsxfun(@times, blkwise_fea, ...
                    2^PCANet.NumFilters(end)./sum(blkwise_fea)); 
           end
           
           Bhist{i} = blkwise_fea;
        end
        
    end           
    f{Idx} = vec([Bhist{:}]');
    
    if ~isempty(PCANet.Pyramid)
        f{Idx} = sparse(f{Idx}/norm(f{Idx}));
    end
end
f = full([f{:}]);

if ~isempty(PCANet.Pyramid)
    BlkIdx = kron((1:size(Bhist{1},1))',ones(length(Bhist)*size(Bhist{1},2),1));
else
    BlkIdx = kron(ones(NumOs,1),kron((1:size(Bhist{1},2))',ones(size(Bhist{1},1),1)));
end

%-------------------------------
function X = Heaviside(X) % binary quantization
X = sign(X);
X(X<=0) = 0;

function x = vec(X) % vectorization
x = X(:);


function beta = spp(blkwise_fea, sam_coordinate, ImgSize, pyramid)

[dSize, ~] = size(blkwise_fea);

img_width = ImgSize(2);
img_height = ImgSize(1);

% spatial levels
pyramid_Levels = length(pyramid);
pyramid_Bins = pyramid.^2;
tBins = sum(pyramid_Bins);

beta = zeros(dSize, tBins);
cnt = 0;

for i1 = 1:pyramid_Levels,
    
    Num_Bins = pyramid_Bins(i1);
    
    wUnit = img_width / pyramid(i1);
    hUnit = img_height / pyramid(i1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(sam_coordinate(1,:) / wUnit);
    yBin = ceil(sam_coordinate(2,:) / hUnit);
    idxBin = (yBin - 1)*pyramid(i1) + xBin;
    
    for i2 = 1:Num_Bins,     
        cnt = cnt + 1;
        sidxBin = find(idxBin == i2);
        if isempty(sidxBin),
            continue;
        end      
        beta(:, cnt) = max(blkwise_fea(:, sidxBin), [], 2);
    end
end


