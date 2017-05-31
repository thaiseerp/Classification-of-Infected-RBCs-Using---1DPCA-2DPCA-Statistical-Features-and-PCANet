function testTrainedPCANetOnSlide()
    %%This function reads each image, Identify patches 32x32 where there is
    %%a chance of parasites (Regional Minima) and then test the location
    %%for the possible parasite and mark it on the slide if it is there. 
    %OK. Run the set up for MATCONVNET ENVIRONMENT
%     setup ;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Get the minima R,G,B across stack
    load('SVMModel6720linear.mat'); load('PCAFilter6720.mat');
    PCANet.NumStages = 2;
    PCANet.PatchSize = [9 9];
    PCANet.NumFilters = [8 8];
    PCANet.HistBlockSize = [9 9]; 
    PCANet.BlkOverLapRatio = 0;
    PCANet.Pyramid = [];
    BasePath = 'EasySeg';
    fnames = dir([BasePath '\ImgesMinAcrsStck\*B.jpg']);
     TPSTV = 0; FPSTV = 0; FNGTV = 0; uWntToCnt = false;
    numfids = length(fnames);  PtchSzM = 32; PtchSzN = 32; 
    prevVidIndx = ''; 
    fh = figure;
    for K = 638:numfids
        display(['Image ' num2str(K)]);
        numPstv = 0; 
        AbsFNme = [BasePath '\ImgesMinAcrsStck\' fnames(K).name]; 
        %Get details needed to acces the Ground Truth file
        [~, FileName, ~] = fileparts(AbsFNme);
        for i = 1:length(FileName); if (FileName(i) == '_'); vid = i-1; break; end; end;
        for j = vid+2:length(FileName); if (FileName(j) == '_'); stck = j-1; break; end; end;
        vidIndx = FileName(1:vid);  stckIndx = FileName(vid+2:stck); 
        load ([BasePath '\GndTrth\' vidIndx '_' stckIndx]);
        if (~strcmp(prevVidIndx, vidIndx)); 
            prevVidIndx = vidIndx;
            cDstLoc = getDstLocnsFor(vidIndx);
        end 
        %Read the image and Make it Single precision
        imgUint8 = imread(AbsFNme); imSingle = rgb2hsv(imgUint8);
        imSingle = single(imSingle(:,:,3)); 
        RC = imgUint8(:, :, 1); GC = imgUint8(:, :, 2); BC = imgUint8(:, :, 3);
        [SzM, SzN, ~] = size(imgUint8); LblMsk = false(SzM, SzN);
        %Compute regional Minima
        regMin = getMyRegionalMinima(imgUint8);
        %Filter out regional minima at the location of WBC s
        
        %If postv to be checkd closer
        chkClsr = cDstLoc & regMin;
        %Now test Each suspected Locns
        Cntrids = regionprops(regMin, 'centroid');
        [numPosns, ~] = size(Cntrids);
        vlidPosCnt = 0;
        for posCnt = 1:numPosns
            
            %For the time being, if there is a patch of the required size
            %around the point, then only we are considering it.
            CCntrids = round(Cntrids(posCnt).Centroid);
            CR = CCntrids(2); CC = CCntrids(1);
            minRw = CR - PtchSzM/2;    minCl = CC - PtchSzN/2;
            maxRw = CR + PtchSzM/2-1;  maxCl = CC + PtchSzN/2-1;
            if (minRw > 0 && minCl > 0 && maxRw <= SzM && maxCl <= SzN) 
                vlidPosCnt = vlidPosCnt + 1;
                %Get curr patch
                currPatch = imSingle(minRw:maxRw, minCl:maxCl, :);
                %Test it and Get the label
%                 
                %Get Feature of the patch
                Feature = PCANet_FeaExt(num2cell(currPatch,[1 2]),PCAFilter6720,PCANet);
                Lbl     = svmclassify(SVMModel6720linear, Feature');
                
                %If the label is postive, closely examine if it is dust?
                if (Lbl == 1)
                    LblMsk(CR, CC) = true;
                end   
            end
        end
        %Now Examine the Msks with the Mask of Dst Pos we have
        LblMskDil       = bwmorph(LblMsk, 'dilate', 3);
        ToBExmnd        = (LblMskDil & chkClsr);
        [~, ToBExmnd]  	= maskAllSharingObjects(LblMskDil, ToBExmnd);
        [~, ToBExmnd]  	= maskAllSharingObjects(LblMsk, ToBExmnd);
        %If the average pixel intensity in the region 7x7 is not below 128
        %Exclude it from Positive
        [clsR, clsC] = find(ToBExmnd);
        for cntCls = 1:length(clsR)
            cClsR = clsR(cntCls); cClsC = clsC(cntCls); 
            patchClse    = GC(cClsR-3:cClsR+3, cClsC-3:cClsC+3);
            meanPatch    = mean(patchClse(:));
            if (meanPatch > 100)
                LblMsk(cClsR , cClsC) = false;
            end
        end
        %Show
        LD = bwmorph(LblMsk, 'dilate', 4);
        RC(LD) = 255; GC(LD) = 0; BC(LD) = 0;
        RGBCmbnd(:, :, 1) = RC; RGBCmbnd(:, :, 2) = GC; RGBCmbnd(:, :, 3) = BC;
        subplot(1, 2, 1); imshow(RGBCmbnd); title('Detected');
        
        MD = bwmorph(Msk, 'dilate', 4);

        RC = imgUint8(:, :, 1); GC = imgUint8(:, :, 2); BC = imgUint8(:, :, 3);
        RC(MD) = 0; GC(MD) = 255; BC(MD) = 0;
        RGBCmbnd2(:, :, 1) = RC; RGBCmbnd2(:, :, 2) = GC; RGBCmbnd2(:, :, 3) = BC;
        subplot(1, 2, 2); imshow(RGBCmbnd2); 
        if (uWntToCnt)
            [TPstv, FNgtv, FPstv] = getCMat(LD, MD, PtchSzM);
            TPSTV = TPSTV+TPstv; FNGTV = FNGTV + FNgtv; FPSTV = FPSTV+FPstv; 
            title(['TPsv = ' num2str(TPstv) ' FNgv = ' num2str(FNgtv) ' FPsv = ' num2str(FPstv)]);
        else
            title('Truth');
        end
        
        print (fh, ['EasySeg\ResultFrmPCAPatch\'  vidIndx '_' stckIndx], '-djpeg');

    end
    
    %Exclude Minima From Bgnd Regions
    
    %Exclude minima in Small objects other than cells?:)
    
    %Get 32x32x3 patch
    
    %Test it with CNN
    
    %Mark the status if parasite.
    [TPSTV,FNGTV,FPSTV]
end

function [TPstv, FNgtv, FPstv] = getCMat(ClsfdD, GnD, PtchSzM)
%     [clearSetLTh, clearSetGTh, remSetGTh] = getBinaryImage(imUint8RGB);
%     fnh = figure;
    [SzM, SzN] = size(GnD);
    HdeMask = false (SzM, SzN);
    HdeMask(PtchSzM/2+1:SzM - PtchSzM/2, PtchSzM/2+1:SzN - PtchSzM/2) = true;
    Trth = GnD & HdeMask; MClsfdD = ClsfdD & HdeMask;
    %Check whether manual intervention is needed or not.
    dilMsk = strel('disk', 17);
    ManualTrth  = imdilate(Trth, dilMsk);      
    NmManTr = bwconncomp(ManualTrth);        NmManTrth = NmManTr.NumObjects;
    ManualClfd = imdilate(MClsfdD, dilMsk);    
    NmManCd = bwconncomp(ManualClfd);        NmManClfd = NmManCd.NumObjects;
    
    
    TrthNumObj    = round(sum(Trth(:))/81);
    ClsfdNmObj    = round(sum(MClsfdD(:))/81);
    if ((NmManTrth ~= TrthNumObj) || (NmManClfd ~= ClsfdNmObj))
       TPstv = input('TPSTV = '); 
       FNgtv = input('FNgtv = ');
       FPstv = input('FPstv = ');
    else
        [~, Mskd] = maskAllSharingObjects(MClsfdD, Trth);
        TPstv      =  round(sum(Mskd(:))/81); %81 pixel for one marking         
        FNgtv       =  TrthNumObj - TPstv;
        FPstv       =  ClsfdNmObj - TPstv; 
    end
end

function susReg = getMyRegionalMinima(imgUint8)
    MnSz = 900;
    hsv = rgb2hsv(imgUint8);
    vlue = hsv(:, :, 3);
    stDsk = strel('disk', 11);
    openimg = imopen(vlue, stDsk);
    mask = imregionalmin(openimg);
    %First Filtering Exclude all the Bgnd
    susReg = (bwareaopen((vlue < graythresh(vlue)), MnSz)) & mask;
end
function cDstLoc = getDstLocnsFor(vidIndx)
    load ('DstLocnsPgm\DstLocByPGM'); cnt = 0;
    vidIndxs = DstLocByPGM.vidIndx;
    [~, numVidIndxs] = size(vidIndxs);
    for i = 1:numVidIndxs
        if (strcmp(vidIndx, vidIndxs(i).name))
            cDstLoc = DstLocByPGM.DstLocByPgm(:, :, i);
            break;
        end
    end
end
function DstLoc = getDustLocns()
close all;
    BasePath = ['E:\Images\GndTruth\Separate\EasySeg\'];
    fnames = dir([BasePath '\ImgesMinAcrsStck\*B.jpg']);
  
    numfids = length(fnames);  M = 32; N = 32; R = 1; Rad = 0.75*M;
    prevVidIndx = ''; DstCnt = 0;
    for K = 388:numfids 
        K
        AbsFNme = [BasePath 'ImgesMinAcrsStck\' fnames(K).name]; 
        %Get details needed to acces the Ground Truth file
        [~, FileName, ~] = fileparts(AbsFNme);
        for i = 1:length(FileName); if (FileName(i) == '_'); vid = i-1; break; end; end;
        for j = vid+2:length(FileName); if (FileName(j) == '_'); stck = j-1; break; end; end;
        vidIndx = FileName(1:vid);  stckIndx = FileName(vid+2:stck); 
        if (strcmp(prevVidIndx, vidIndx)); continue; end
        prevVidIndx = vidIndx; 
        figure; imshow(imread(AbsFNme)); title(num2str(K));
        ImgStack = getImageStackFromVideoIndx(vidIndx);
        DstCnt = DstCnt+1;
        DstLocByPgm = computeDustLocFrmStk(ImgStack);
        save (['DstLocnsPgm\DstLocByPGM_' vidIndx], 'DstLocByPgm');
        if (strcmp(vidIndx, 'I') || strcmp(vidIndx, 'J'))
            stp =1;
        end        
    end
%      getDstLocByMajority();
end
function  DstLocByPGM = getDstLocByMajority()
    BasePath = ''; %['E:Images\GndTruth\Separate\EasySeg\'];
    fnames = dir([BasePath 'DstLocnsPgm\DstLocByPGM_*.mat']);
  
    numfids = length(fnames);  M = 32; N = 32; R = 1; Rad = 0.75*M;
    VoteSecnd = uint8(zeros(480, 720));
    VoteFirst = uint8(zeros(480, 720));
    KTop = [25 27 29 31 33];
    close all;
    for K = 1:numfids 
        AbsFNme = [BasePath 'DstLocnsPgm\' fnames(K).name]; 
        [~, FileName, ~] = fileparts(AbsFNme);
        load (AbsFNme);
        vidIndx = FileName(13:end);
        [SzM, SzN] = size(DstLocByPgm);
        if (K == 26 || K == 28 || K == 30 || K == 32 || K == 34)
%             vidIndx
            VoteSecnd(DstLocByPgm) =  VoteSecnd(DstLocByPgm)+1;
%             figure; imshow(VoteSecnd, []); 
        else
            VoteFirst(DstLocByPgm) = VoteFirst(DstLocByPgm)+1;
%             figure; imshow(VoteFirst, []); 
        end
        display([num2str(K) '   ' vidIndx '   ' num2str(SzM) '    ' num2str(SzN)]);
    end
    DstLocByPgmFirst = VoteFirst > 15;
    DstLocByPgmSecond = VoteSecnd > 3;
%     DstLocByPgm = false(480, 720);
    for K = 1:numfids
        AbsFNme = [BasePath 'DstLocnsPgm\' fnames(K).name]; 
        [~, FileName, ~] = fileparts(AbsFNme);
        vidIndx = FileName(13:end);
        DstLocByPGM.vidIndx(K).name = vidIndx;
        if (K == 26 || K == 28 || K == 30 || K == 32 || K == 34)
            vidIndx
            DstLocByPGM.DstLocByPgm(:, :, K) = DstLocByPgmSecond;
        else
            DstLocByPGM.DstLocByPgm(:, :, K) = DstLocByPgmFirst;
        end
    end
    DstLocByPGM.meta = 'Genrtd By PGM testTrainedCNNNetOnSlide in MatCNetMalClass Fldr';
    save('DstLocnsPgm\DstLocByPGM', 'DstLocByPGM');
end
function DstLocByPgm = computeDustLocFrmStk(ImgStackUint8)
close all; DstLocByPgm = 0;
    [SzM, SzN, SzO, numCells] = size(ImgStackUint8);
    numAssns = zeros(SzM, SzN);
    Addns = zeros( SzM, SzN);
    currImD = 0; Diffr = zeros( SzM, SzN);
    objsIntst = false(SzM, SzN, numCells);
    Vote = uint8(zeros(SzM, SzN));
    close all;
    for i = 1:numCells
        imUint8 = ImgStackUint8(:, :, :, i);
        [clearSetLTh, clearSetGTh, remSetGTh] = getBinaryImage(imUint8);
        segImg = clearSetLTh | clearSetGTh | remSetGTh;
        bgndImg = ~segImg;
%         figure; imshow(bgndImg);
        bgndErde = bwmorph(bgndImg, 'erode', 25);
        G = imUint8(:, :, 2);
        avgbgnd = mean(G(bgndErde));
        objsCand = G < (avgbgnd - 0.05*avgbgnd);
        %remove all bigger ones
        intstd = objsCand & ~bwareaopen(objsCand, 200);
        objsIntst(:, :, i) = intstd;
        Vote(intstd) = Vote(intstd)+1;
%         figure; imshow(imUint8); title(num2str(i));
%         figure; imshow(intstd);
    end
    %Select those having 25% Support
    mnSprt = 0.25; %support
    DstLocByPgm = Vote > max(round(mnSprt*numCells), 4);
%     figure; imshow(imUint8);
%     figure; imshow(fnalDstLocs);
end
function [clearSetLTh, clearSetGTh, remSetGTh] = getBinaryImage(imUint8RGB)
    imG     = im2double(imUint8RGB(:, :, 2));
    msk     = fspecial('average', 15);
    avG     = imfilter(imG, msk);
    fildLThImg = bwfill(bwareaopen(imG < (avG - 0.01), 200), 'holes');
    fildGThImg = lOtThresh(imUint8RGB);
    [clearSetLTh, ~] = getClearSet(fildLThImg);
    remSetGTh = bwareaopen(bwmorph(~clearSetLTh & fildGThImg, 'open', 3), 200);
    [clearSetGTh, remSetGTh] = getClearSet(remSetGTh);
end

function [clearSet, remSet] = getClearSet(bwImg)
    remSet = bwImg; clearSet = false(size(remSet));
    cmps = regionprops(bwImg, {'PixelIdxList', 'Solidity', 'ConvexImage', 'BoundingBox'});
    [numObj, ~] = size(cmps);
    lowThresh = 750; highThresh = 2000; 
     for i = 1:numObj
        currObj = cmps(i).PixelIdxList;
        objArea = length(currObj);
        if (cmps(i).Solidity > 0.9 && objArea > lowThresh && objArea < highThresh)
            clearSet(currObj) = true;
            remSet(currObj) = false;
        end
    end   
end

function ThImg = lOtThresh(im)
    div = 2;
    [SzM, SzN, SzO] = size(im);
    if (SzO > 3)
        im = rgb2gray(im); %im(:, :, 2);
    end
    im = im2double(im);
    rOfst = round(SzM/div); cOfst = round(SzN/div);
    ThImg = false(SzM, SzN);
    for i = 1:div
        startR = (i-1)*rOfst+1;
        if (i == div)
            endR = SzM;
        else
            endR  = i*rOfst;
        end
        for j = 1:div
            startC = (j-1)*cOfst+1;
            if (j == div)
                endC = SzN;
            else
                endC  = j*cOfst;
            end
            divImg = im(startR:endR, startC:endC);
            ThImg(startR:endR, startC:endC) = divImg < (graythresh(divImg)+0.01);
       end
    end
    ThImg = bwfill(bwareaopen(ThImg, 200), 'holes');
end

function [remMsk, Mskd] = maskAllSharingObjects(BaseImg, ShareObjs)
    ToBMaskd    = BaseImg & ShareObjs;
    LblsFrmBase = bwlabel(BaseImg);
    Mskd = false(size(BaseImg));
    lbls2BMaskd = LblsFrmBase(ToBMaskd);
    unqLbls     = unique(lbls2BMaskd(:));
    for lbl = 1:length(unqLbls)
        Mskd(LblsFrmBase  == (unqLbls(lbl))) = true;
    end
    remMsk = BaseImg & ~Mskd;  
end

function ImgStack = getImageStackFromVideoIndx(vidIndx)
    BasePath = ['E:Images\GndTruth\Separate\EasySeg\'];
    fnames = dir([BasePath '\ImgesMinAcrsStck\' vidIndx '*B.jpg']);
  
    numfids = length(fnames);  M = 32; N = 32; R = 1; Rad = 0.75*M;
    for K = 1:numfids 
        AbsFNme = [BasePath 'ImgesMinAcrsStck\' fnames(K).name]; 
        im = imread(AbsFNme);
        ImgStack(:, :, :, K) = im;
    end
end