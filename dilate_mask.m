function x = dilate_mask(mask)
    temp=fspecial("gaussian",35,35/3);
    #imshow(temp)
    y=im2double(mask);
    y=imfilter(y,temp);
    y(y>=0.1)=1;
    x=im2uint16(y);
    
end