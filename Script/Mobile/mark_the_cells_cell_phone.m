clear;
load('im1_data');
im1_data = train_data;
load('im2_data');
im2_data = train_data;
load('im3_data');
im3_data = train_data;
load('im4_data');
im4_data = train_data;
load('im5_data');
im5_data = train_data;
load('im6_data');
im6_data = train_data;

MARGIN = [-1.5 -1.5 3 3];
im1_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series009_z0_ch02.tif';
im2_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series028_z0_ch02.tif';
im3_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series033_z0_ch02.tif';
im4_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series055_z0_ch02.tif';
im5_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series059_z0_ch02.tif';
im6_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series095_z0_ch02.tif';

im1_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series009_z0_ch02_Co.txt';
im2_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series028_z0_ch02_Co.txt';
im3_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series033_z0_ch02_Co.txt';
im4_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series055_z0_ch02_Co.txt';
im5_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series059_z0_ch02_Co.txt';
im6_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series095_z0_ch02_Co.txt';

save_names = {'Series009_1000', 'Series028_100', 'Series033_1000', ...
    'Series055_100', 'Series059_1000', 'Series095_100'};

im_paths = {im1_path, im2_path, im3_path, ...
    im4_path, im5_path, im6_path};

coord_paths = {im1_coord_path, im2_coord_path, im3_coord_path, ...
    im4_coord_path, im5_coord_path, im6_coord_path};

label_data = {im1_data, im2_data, im3_data, im4_data, ...
    im5_data, im6_data};


for i = 1 : 6 
    %i = 1; % temporary
    im = imread(im_paths{i});
    level = graythresh(im);
    level = 0.5*level;  % 0.6, 0.7
    bw_im = im2bw(im, level);
    points = regionprops(bw_im, im, 'BoundingBox', 'Area', 'Centroid', ...
    'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Perimeter', ...
    'Solidity', 'PixelIdxList', 'PixelList', 'MaxIntensity', 'MinIntensity', ...
    'WeightedCentroid', 'BoundingBox', 'MeanIntensity', 'Orientation', ...
    'Extrema', 'Extent', 'ConvexHull');
    % concatenates bounding boxes of all points; prepare for insertShape
    load('im1_data');
    predictions = label_data{i}(:, 1);
    positions = zeros(length(points), 4); %(x, y, width, height)
    for k = 1 : length(points)
        if predictions(k) == 1
            positions(k, :) = points(k).BoundingBox + MARGIN;
        end
    end

    
    fileID = fopen(coord_paths{i});
    C = textscan(fileID,'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter', '\t');
    labels = zeros(length(C{4}), 4);
    C4 = char(C{4});
    C5 = char(C{5});
    C6 = char(C{6});
    C7 = char(C{7});
    labels(:, 1) = str2num(C4);
    labels(:, 2) = str2num(C5);
    labels(:, 3) = str2num(C6);
    labels(:, 4) = str2num(C7);
    labeled_image = insertShape(im, 'Rectangle', labels);
    labeled_image_2 = insertShape(im, 'Rectangle', positions, 'Color', 'red');
    
    figure;
    imshow(labeled_image);
    handle = figure;
    imshow(labeled_image_2);
    saveas(handle, [save_names{i} '.fig']);  
    % image must be open in order to save
end

