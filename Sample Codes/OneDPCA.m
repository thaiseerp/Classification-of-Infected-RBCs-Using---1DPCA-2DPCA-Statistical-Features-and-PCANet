function [EigenVectors] = OneDPCA(data)
    z = zeros(size(data,1),size(data,2));
    s = zeros(size(data,2));
    for i = 1:size(data,1)
        z(i,:) = mean(data) - data(i,:);
        s = s + z(i,:)'*z(i,:);
    end
    [EigenVec,EigenVal] = eig(s);
    [~,indx] = sort(max(EigenVal));
    indx = flip(indx);
    
    EigenVectors = zeros(size(EigenVec,1));
    for i = 1:size(indx,2)
    EigenVectors(:,i) = EigenVec(:,indx(1,i));
    end
end