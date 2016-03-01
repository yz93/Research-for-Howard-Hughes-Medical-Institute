im1 = imread('Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_1_R2.tif');
im2 = imread('Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_4_R2.tif');
im3 = imread('Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_6_R2.tif');
im1_o = imread('Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_1_R2.tif');
im2_o = imread('Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_4_R2.tif');
im3_o = imread('Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_6_R2.tif');
% 1100 x 1100 crop for all 3
im1_crop = imcrop(im1_o, [1330, 1370, 1100, 1100]);
im2_crop = imcrop(im2_o, [1292, 1590, 1100, 1100]);
im3_crop = imcrop(im3_o, [1426, 899, 1100, 1100]);

handle1=figure;
imshow(im1_crop);
imsave(handle1);

handle2=figure;
imshow(im2_crop);
imsave(handle2);

handle3=figure;
imshow(im3_crop);
imsave(handle3);

