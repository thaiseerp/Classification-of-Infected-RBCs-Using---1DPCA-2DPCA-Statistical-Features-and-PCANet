function StatisticalFeatures()
tic;
    FeatMal = getMyDataset([60 20 20],[1.1652 0.3895 98.4453]);
   % ConfMatSVM = SVMClassification(FeatMal);
    ConfMatNN = FFNNBPClassification(FeatMal, 20);
   % getStatistics(ConfMatSVM);
    getStatistics(ConfMatNN);
toc;
end
function imdb = getMyDataset(PercPstvTrainVaidnTest, PercNgtvTrainVaidnTest)   
    
    load ('FeatMalClAsWhleExcldDstCl');
    Lbl = FeatMal.feature.label;
    FeatPstv = FeatMal.feature.data(Lbl == 1,:);  FeatPstv(:, 2) = [];
    FeatNgtv = FeatMal.feature.data(Lbl == 2, :); FeatNgtv(:, 2) = [];
    imdb.features.cellId = FeatMal.feature.cellIds;

    numPstvImgs             = size(FeatPstv, 1);
    numNgtvImgs             = size(FeatNgtv, 1);
    pstvSplit               = divideDataInRatio(numPstvImgs, PercPstvTrainVaidnTest);  
    ngtvSplit               = divideDataInRatio(numNgtvImgs, PercNgtvTrainVaidnTest);
    imdb.features.data      = [FeatPstv; FeatNgtv];
    imdb.features.label     = [ones(numPstvImgs, 1); 2*ones(numNgtvImgs, 1)];
    imdb.features.set       = [pstvSplit ngtvSplit];
    imdb.features.id        = 1:(numPstvImgs+numNgtvImgs);
    imdb.meta.classes       = 'Malaria,Healthy';
    imdb.meta.sets          = {'train', 'val', 'test'};
    
end

function TnTtVn = divideDataInRatio(MaxLt, Ratio)
    blck    = randperm(MaxLt); strt = 1;
    cmSum   = cumsum(Ratio);
    TnTtVn  = zeros(1, MaxLt);
    for i = 1:length(Ratio)
        intstd      = round(cmSum(i)/100*MaxLt);
        TnTtVn((blck >= strt) &  (blck <= intstd)) = i;
        strt        = intstd+1;
    end   
end

function DvdData = DivideDataset(MalImdb)
    % Training set
    TrIndx         = (MalImdb.features.set == 1);
    DvdData.TrLbl    = MalImdb.features.label(TrIndx);
    DvdData.TrClsId  = MalImdb.features.cellId(TrIndx);
    TrFeat           = MalImdb.features.data(TrIndx, :);
    
    % Validaton Set
    ValIndx         = (MalImdb.features.set == 2);
    DvdData.ValLbl   = MalImdb.features.label(ValIndx);
    DvdData.ValClsId = MalImdb.features.cellId(ValIndx);
    ValFeat           = MalImdb.features.data(ValIndx, :);
    
    % Test Set
    TsIndx          = (MalImdb.features.set == 3);
    DvdData.TsLbl     = MalImdb.features.label(TsIndx);
    DvdData.TsClsId   = MalImdb.features.cellId(TsIndx);
    TsFeat            = MalImdb.features.data(TsIndx, :);
    
    
    % Subtract Mean of Train features from all
    featMean         = mean(TrFeat);
    DvdData.TrFeat   = (TrFeat - repmat(featMean, size(TrFeat, 1), 1)); 
    DvdData.ValFeat   = (ValFeat - repmat(featMean, size(ValFeat, 1), 1)); 
    DvdData.TsFeat    = TsFeat  - repmat(featMean, size(TsFeat, 1), 1); 
end

function ConfnMatSVM = SVMClassification(MalImdb)
    DvdData = DivideDataset(MalImdb);
    SVMModel    = svmtrain(DvdData.TrFeat, DvdData.TrLbl);% 'method', 'SMO', 'kernel_function', 'rbf');
    
    PrTrLbl       = svmclassify(SVMModel, DvdData.TrFeat);
    ConfnMatTr  = confusionmat(DvdData.TrLbl,PrTrLbl);
    
    PrValLbl      = svmclassify(SVMModel, DvdData.ValFeat);
    ConfnMatVal  = confusionmat(DvdData.ValLbl,PrValLbl); %getConfnMatx(LblVn, DvdData.validnLbl);
    
    PrTsLbl       = svmclassify(SVMModel, DvdData.TsFeat); % getClsfdLblsInChunks
    ConfnMatTs  = confusionmat(DvdData.TsLbl,PrTsLbl); %getConfnMatx(PrTsLbl, DvdData.testLbl); 
    
    ConfnMatSVM  = [ConfnMatTr; ConfnMatVal; ConfnMatTs];
end

function ConfnMatNN = FFNNBPClassification(MalImdb, HNSize)
    DvdData = DivideDataset(MalImdb);

    TrData  = DvdData.TrFeat';  TrLabel = DvdData.TrLbl';
    ValData  = DvdData.ValFeat';  ValLabel = DvdData.ValLbl';
    TsData  = DvdData.TsFeat';   TsLabel = DvdData.TsLbl';
    
    numClasses = max(TrLabel);
    trLabels = zeros(numClasses, size(TrData,2));
    valLabel = zeros(numClasses, size(ValData,2));
    tsLabel = zeros(numClasses, size(TsData,2));
    for j = 1:numClasses
        trLabels(j, TrLabel == j) = 1;
        valLabel(j, ValLabel == j) = 1;
        tsLabel(j, TsLabel == j) = 1;
    end
    
    % Neural Network Starts Here
    net = patternnet(HNSize);
    net.trainParam.showWindow = false;
    [net, ~] = train(net,TrData,trLabels);
    
    
    PrTrLbl       = net(TrData);
    [~,ConfnMatTr,~,~] = confusion(trLabels,PrTrLbl);
    
    PrValLbl       = net(ValData);
    [~,ConfnMatVal,~,~] = confusion(valLabel,PrValLbl);
    
    PrTsLbl       = net(TsData);
    [~,ConfnMatTs,~,~] = confusion(tsLabel,PrTsLbl);
 
    ConfnMatNN  = [ConfnMatTr; ConfnMatVal; ConfnMatTs];
end

function [Sensitivity, Specificity, FScore] = getStatistics(ConfnMatxs)
    Sensitivity = zeros(3, 1); Specificity = zeros(3, 1); FScore = zeros(3, 1);
    for i = 1:3
        CConfnMat = ConfnMatxs((i-1)*2+1:i*2, 1:2);
        Sensitivity(i) = CConfnMat(1, 1)/(CConfnMat(1, 1) + CConfnMat(1, 2));
        Specificity(i) = CConfnMat(2, 2)/(CConfnMat(2, 2) + CConfnMat(2, 1));
        FScore(i) = (2*CConfnMat(1, 1))/((2*CConfnMat(1, 1)) + CConfnMat(1, 2) + CConfnMat(2, 1));
        disp(CConfnMat);
        fprintf('\n Sensitivity is %f \n',Sensitivity(i));
        fprintf('Specificity is %f \n',Specificity(i));
        fprintf('FScore is %f \n',FScore(i));
    end
end