function Images = Img2Cell(Image)
Images = cell(1,size(Image,3));
for i=1:size(Image,3)
    Images{i} = Image(:,:,i);
end