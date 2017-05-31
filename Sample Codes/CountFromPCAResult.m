function count = CountFromPCAResult()
    TPSTV = 0; FPSTV = 0; FNGTV = 0;
    
    BasePath = 'EasySeg';
    fnames = dir([BasePath '\ResultFrmPCAPatch\*.jpg']);
    
    numfids = length(fnames);  PtchSzM = 32; PtchSzN = 32;
    
    for K = 1:numfids
        AbsFNme = [BasePath '\ResultFrmPCAPatch\' fnames(K).name];
        [~, FileName, ~] = fileparts(AbsFNme);
        
        load ([BasePath '\GndTrth\' FileName]);
        MD = bwmorph(Msk, 'dilate', 4);
        
        imgUint8 = imread(AbsFNme); imSingle = rgb2hsv(imgUint8);
        imSingle = single(imSingle(:,:,3));
        
        [SzM, SzN, ~] = size(imgUint8); DetMsk = false(SzM, SzN);
        
        
        if(red)
            CR = Rrow;
            CC = Rcol;
            DetMsk(CR, CC) = true;
        end
        
        
        
        count = getCMat(DetMsk, MD, PtchSzM);
    end  
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