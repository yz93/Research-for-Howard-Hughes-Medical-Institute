%clc  % Clear the command window.
%close all; % Close all figures 
clear; % Erase all existing variables.
im1_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series009_z0_ch02.tif';
im1_lb_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series009_z0_ch02_237.tif';
im1_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series009_z0_ch02_Co.txt';

im2_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series028_z0_ch02.tif';
im2_lb_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series028_z0_ch02_170.tif';
im2_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series028_z0_ch02_Co.txt';

im3_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series033_z0_ch02.tif';
im3_lb_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series033_z0_ch02_188.tif';
im3_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series033_z0_ch02_Co.txt';

im4_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series055_z0_ch02.tif';
im4_lb_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series055_z0_ch02_228.tif';
im4_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series055_z0_ch02_Co.txt';

im5_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series059_z0_ch02.tif';
im5_lb_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series059_z0_ch02_184.tif';
im5_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series059_z0_ch02_Co.txt';

im6_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series095_z0_ch02.tif';
im6_lb_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series095_z0_ch02_238.tif';
im6_coord_path = 'Z:\Winter 2016\HHMI\20160107RCP_Microscope\Data Sets\Series095_z0_ch02_Co.txt';

im_paths = {im1_path, im2_path, im3_path, im4_path, im5_path, im6_path};
im_coord_paths = {im1_coord_path, im2_coord_path, im3_coord_path,...
    im4_coord_path, im5_coord_path, im6_coord_path};
data_file_names = {'im1_data', ...
    'im2_data', ...
    'im3_data', ...
    'im4_data', ...
    'Z:\Winter 2016\HHMI\20160107RCP_Microscope\New_Trial\im5_data', ...
    'Z:\Winter 2016\HHMI\20160107RCP_Microscope\New_Trial\im6_data'};
%im_path = im6_path;  %im5_path; %im4_path;  %im3_path;%im2_path;  %im1_path;
%im_coord_path = im6_coord_path;  %im5_coord_path;  %im4_coord_path;  %im3_coord_path;%im2_coord_path;  %im1_coord_path;
%data_file_name = 'im6_data';  %'im5_data';  %'im4_data';  %'im3_data';%'im2_data';  %'im1_data';

for x = 1 : 6
    fileID = fopen(im_coord_paths{x});
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
    label_all_pixels = zeros(2, 2);   %(width, length) don't know exact length, so allocate 2 first.
    c = 1;
%     quarter_start = round(labels(i, 3)/4);
%     quarter_end = round(3*labels(i, 3)/4);
%     c_quarter_start = round(labels(i, 4)/4);
%     c_quarter_end = round(3*labels(i, 4)/4);
    for i = 1 : length(labels)
        for j = round(labels(i, 3)/4) : labels(i, 3)
            for k = round(labels(i, 4)/4) : labels(i, 4)
                label_all_pixels(c, 1) = labels(i, 1) + k;
                label_all_pixels(c, 2) = labels(i, 2) + j;  % width
                c = c + 1;
            end
        end
    end

    im = imread(im_paths{x});
    level = graythresh(im);
    if (x == 4)
        %im = medfilt2(im,[3 3]);
        %level = graythresh(im);
        level = 0.7 * level;
    else
        level = 0.6*level;  % 0.6, 0.7
    end
    bw_im = im2bw(im, level);
%      figure;
%      imshow(im);
%     figure;
%     imshow(bw_im);
    
    points = regionprops(bw_im, im, 'Area','Centroid','Eccentricity','MajorAxisLength','MinorAxisLength','Perimeter','Solidity','PixelIdxList','PixelList','MaxIntensity','MinIntensity','WeightedCentroid','BoundingBox','MeanIntensity','Orientation','Extrema','Extent','ConvexHull');%,'MeanIntensity','Orientation'
    train_data = zeros(length(points), 16);
    centroids = cat (1, points.Centroid);
    for i = 1 : length(points) % write out the list of possible objects and their properties
        weighted_centroids = cat(1, points.WeightedCentroid);
        train_data (i,2) = double(points(i).Area);
        train_data (i,3) = double(centroids(i,1));
        train_data (i,4) = double(centroids(i,2));
        train_data (i,5) = double(weighted_centroids(i,1));
        train_data (i,6) = double(weighted_centroids(i,2)); 
        train_data(i,7) = size(points(i).ConvexHull, 1); % number of vertex of the smallest convex polygon that can contain the region
        train_data (i,8) = double(points(i).Area)/size(points(i).ConvexHull,1);%
        train_data (i,9) = double(points(i).Solidity); % Area/ConvexArea
        train_data (i,10)= double(points(i).Extent); % Area divided by the area of the bounding box
        train_data (i,11) = double(points(i).MeanIntensity);
        train_data (i,12) = double(points(i).MaxIntensity);
        train_data (i,13) = double(points(i).MinIntensity);
        train_data (i,14) = sqrt((train_data(i,3)-train_data(i,5))^2 + (train_data(i,4)-train_data(i,6))^2);
        train_data (i,15) = double(points(i).Perimeter)/double(points(i).Area); %Perimeter/Area;
        train_data (i,16) = double(points(i).MaxIntensity)-double(points(i).MinIntensity);%MaxIntensity subtract MinIntensity
    %==========Code for Labeling=============
    % for each point, check if the (x, y) coordinate of the label is included in its PixelList matrix.
        key_pixel = intersect(points(i).PixelList, label_all_pixels, 'rows');
        if (numel(key_pixel) >= 1)
            train_data (i,1) = 1;
        end
    end

    % figure(7); imshow(bw_im);
    % hold on
    % [row, col] = find(train_data(:, 1));
    % plot (centroids(row,1), centroids(row,2), 'y*');
    % text(centroids(row,1), centroids(row,2), num2str(row),'horizontal','center', 'vertical','middle','BackgroundColor', [1 1 1]);     
    % hold off
    % 
    % figure(8); imshow(im_lb);

    save(data_file_names{x}, 'train_data');
end
%===========Machine Learning============
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
final_train_data = [im1_data; im2_data; im3_data; im4_data; im5_data; im6_data];
%=============Cross Validation -- Bagging=============
bag_accu = zeros(1, 6);
fake_bag_accu = zeros(1, 6);
bag_accu_per_point = zeros(1, 6);
count_original = [237, 170, 188, 228, 184, 238];
for i = 1 : 6
    if (i == 1)
        s_validate = 1;
        e_validate = length(im1_data);
    elseif (i == 2)
        s_validate = length(im1_data) + 1;
        e_validate = s_validate + length(im2_data) - 1;
    elseif (i == 3)
        s_validate = length(im1_data) + length(im2_data) + 1;
        e_validate = s_validate + length(im3_data) - 1;
    elseif (i == 4)
        s_validate = length(im1_data) + length(im2_data) + length(im3_data) + 1;
        e_validate = s_validate + length(im4_data) - 1;
    elseif (i == 5)
        s_validate = length(im1_data) + length(im2_data) + ...
            length(im3_data) + length(im4_data) + 1;
        e_validate = s_validate + length(im5_data) - 1;
    elseif (i == 6)
        s_validate = length(im1_data) + length(im2_data) + ...
            length(im3_data) + length(im4_data) + length(im5_data) + 1;
        e_validate = s_validate + length(im6_data) - 1;
    end
    validate_data = final_train_data(s_validate : e_validate, :);
    if (i == 1)
        train = final_train_data(length(im1_data) + 1 : end, :);
    elseif (i == 6)
        train = final_train_data(1 : length(final_train_data)-length(im6_data), :);
    else
        train = [final_train_data(1 : s_validate-1, :); final_train_data(e_validate+1 : end , :)];
    end
    % final_trainingdata includes labels at the last column
    ens_classifier = fitensemble(train(:, 2:end), train(:, 1), 'bag', 1000, 'Tree', 'type', 'classification');
    ens_predictions = predict(ens_classifier, validate_data(:, 2:end));
    bag_accu(i) = 1 - abs(nnz(ens_predictions) / count_original(i) - 1);
    fake_bag_accu(i) = 1 - abs(nnz(ens_predictions) / nnz(validate_data(:, 1)) - 1);
    bag_accu_per_point(i) = 1 - nnz(ens_predictions - validate_data(:, 1)) / length(validate_data(:, 1));
end

bag_accu

bag_accu_per_point


