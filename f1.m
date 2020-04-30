function f1(styleimgname,objimgname,ssize,osize,oscale,opos)
  pkg load image
  [objimg, omap, oalpha]=imread(objimgname);
  [styleimg, smap, salpha]=imread(styleimgname);
  osize=[osize(2) osize(1)];
  ssize=[ssize(2) ssize(1)];
  opos=[opos(2) opos(1)];
  if(size(salpha)==[0 0])
    salpha=255*ones(ssize);
  endif
  objimg=imresize(objimg,osize);
  styleimg=imresize(styleimg,ssize);
  oalpha=imresize(oalpha,osize);
  salpha=imresize(salpha,ssize);
  objimg=imresize(objimg,oscale);
  oalpha=imresize(oalpha,oscale);
  
  mask=zeros(ssize);
  # objimg(:,:,1)=colortransfer(objimg(:,:,1),styleimg(:,:,1),oalpha);
  # objimg(:,:,2)=colortransfer(objimg(:,:,2),styleimg(:,:,2),oalpha);
  # objimg(:,:,3)=colortransfer(objimg(:,:,3),styleimg(:,:,3),oalpha);
  
  p1img=styleimg;

  osize=size(oalpha);
  opos=floor(opos);
  pos=ssize(1)-osize(1)-opos(1);
  save mydata oalpha;
  for i=1:osize(1)
    for j=1:osize(2)
      al=oalpha(i,j);
      x=(int32(al)*int32(objimg(i,j,1)) + int32(255-al)*int32(styleimg(pos+i,opos(2)+j,1)))*inverse(255);
      y=(int32(al)*int32(objimg(i,j,2)) + int32(255-al)*int32(styleimg(pos+i,opos(2)+j,2)))*inverse(255);
      z=(int32(al)*int32(objimg(i,j,3)) + int32(255-al)*int32(styleimg(pos+i,opos(2)+j,3)))*inverse(255);
      p1img(pos+i,opos(2)+j,1)=floor(uint8(x));
      p1img(pos+i,opos(2)+j,2)=floor(uint8(y));
      p1img(pos+i,opos(2)+j,3)=floor(uint8(z));
      if al>100
        mask(pos+i,opos(2)+j)=1;
      endif
    endfor
  endfor
  load present.mat;
  present=present+1;
  save present.mat present;
  dmask=dilate_mask(mask);
  imwrite(p1img,"newimg.png","Alpha",salpha);
  imwrite(mask,["outputs/" num2str(present) "_mask.png"]);
  imwrite(dmask,["outputs/" num2str(present) "_mask_dilated.png"]);
  imwrite(p1img,["outputs/" num2str(present) "_input.png"]);
  imwrite(styleimg,["outputs/" num2str(present) "_style.png"]);

  

endfunction
