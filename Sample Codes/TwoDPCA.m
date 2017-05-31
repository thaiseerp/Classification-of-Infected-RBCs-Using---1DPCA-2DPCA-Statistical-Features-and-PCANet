function [EigenVectors] = TwoDPCA(data)
    ImageSum = zeros(size(data,1),size(data,2));
    for i=1:size(data,3)
        ImageSum = ImageSum + data(:,:,i);
    end
    ImageAvg = ImageSum/size(data,3);
    
    G = zeros(size(data,2));
    for i=1:size(data,3)
        G = G + (data(:,:,i)-ImageAvg)'*(data(:,:,i)-ImageAvg);
    end
    G = G/size(data,3);
    [EigenVec,EigenVal] = eig(G);
    [~,indx] = sort(diag(EigenVal));
    indx = flip(indx');
    
    EigenVectors = zeros(size(EigenVec,1));
    for i = 1:size(indx,2)
    EigenVectors(:,i) = EigenVec(:,indx(1,i));
    end
end