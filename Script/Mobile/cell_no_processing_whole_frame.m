close all; % Close all figures 
clear;  % Erase all existing variables.
im1_path = 'Z:\Winter 2016\HHMI\cellphone 1_1000\Mix1to1000_1_R.tif';
%'Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_1_R2.tif';
%'Z:\Winter 2016\HHMI\cellphone 1_1000\Mix1to1000_1_R.tif';
im1_coord_path = 'Z:\Winter 2016\HHMI\cellphone 1_1000\red_data_summary\Mix1to1000_1_R_ROI_594.txt';

im2_path = 'Z:\Winter 2016\HHMI\cellphone 1_1000\Mix1to1000_4_R.tif';
%'Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_4_R2.tif';
% 'Z:\Winter 2016\HHMI\cellphone 1_1000\Mix1to1000_4_R.tif';
im2_coord_path = 'Z:\Winter 2016\HHMI\cellphone 1_1000\red_data_summary\Mix1to1000_4_R_ROI_563.txt';

im3_path = 'Z:\Winter 2016\HHMI\cellphone 1_1000\Mix1to1000_6_R.tif';
%'Z:\Winter 2016\HHMI\RCP_1to1000\Mix1to1000_6_R2.tif';
% 'Z:\Winter 2016\HHMI\cellphone 1_1000\Mix1to1000_6_R.tif';
im3_coord_path = 'Z:\Winter 2016\HHMI\cellphone 1_1000\red_data_summary\Mix1to1000_6_R_ROI_737.txt';

im_paths = {im1_path, im2_path, im3_path};
coord_paths = {im1_coord_path, im2_coord_path, im3_coord_path};

for_level = [3.1, 3.1, 3.2];
for_bareopen = [12, 12, 6];

num_identified_labels = zeros(1, 3);

% _2 is original images without processing from Qingshan
save_names = {'mobile_im1_data_2', 'mobile_im2_data_2', 'mobile_im3_data_2'};
for x = 1 : 3
    fileID = fopen(coord_paths{x});
    C = textscan(fileID,'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter', '\t');
    labels = zeros(length(C{4}), 4);
    C4 = char(C{4});
    C5 = char(C{5});
    C6 = char(C{6});
    C7 = char(C{7});
    labels(:, 1) = str2num(C4); %#ok<*ST2NM>
    labels(:, 2) = str2num(C5);
    labels(:, 3) = str2num(C6);
    labels(:, 4) = str2num(C7); 
%     label_all_pixels = zeros(2, 2);   %(width, length) don't know exact length, so allocate 2 first.
%     c = 1;
%     for i = 1 : length(labels)
%         for j = 0 : labels(i, 3)
%             for k = 0 : labels(i, 4)
%                 label_all_pixels(c, 1) = labels(i, 1) + k;
%                 label_all_pixels(c, 2) = labels(i, 2) + j;  % width
%                 c = c + 1;
%             end
%         end
%     end
    im = imread(im_paths{x});
    im = im2uint8(im);
    im = medfilt2(im,[3 3]);
    im = imadjust(im, [0.01 0.9], []);
    %figure; 
    %imshow(im);
    level = graythresh(im);
    %level = multithresh(im);
    %seg_I = imquantize(im,level);
    %figure;
    %imshow(seg_I);
    bw_im = im2bw(im, for_level(x)*level);  % 1.8 works great
    %temp_mask = bwareaopen(bw_im, 150);
    %bw_im = bw_im - temp_mask;
    bw_im = bwareaopen(bw_im, for_bareopen(x));
    %figure; 
    %imshow(bw_im);  %im, 
    CC = bwconncomp(bw_im);
    points = regionprops(CC, im, 'Area', 'Centroid', ...
    'Eccentricity', 'MajorAxisLength', 'MinorAxisLength', 'Perimeter', ...
    'Solidity', 'PixelIdxList', 'PixelList', 'MaxIntensity', 'MinIntensity', ...
    'WeightedCentroid', 'BoundingBox', 'MeanIntensity', 'Orientation', ...
    'Extrema', 'Extent', 'ConvexHull');
    train_data = zeros(length(points), 16);
    centroids = cat (1, points.Centroid);
    for i = 1 : length(points) % write out the list of possible objects and their properties
        %dynamic_labels = labels;
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
    % for each point, check if the (x, y) coordinate of the label is included
    % in its PixelList matrix.
        for j = 1 : length(labels)
            if (rectint(points(i).BoundingBox, labels(j, :)) ~= 0)
                train_data (i,1) = 1;
                break;
                %labels = removerows(dynamic_labels,'ind', i);  
                %dynamic_len = dynamic_len - 1;
                % so if there is duplicate centroids for one label, only
                % one is counted. A makeshift way to avoid overcounting 
                % candidates.
            end
        end
%         key_pixel = intersect(points(i).PixelList, label_all_pixels, 'rows');
%         if (numel(key_pixel) >= 1)
%             train_data (i,1) = 1;
%         end
    end
    num_identified_labels(x) = nnz(train_data(:, 1));
    save(save_names{x}, 'train_data');
end
% figure(7); imshow(bw_im);
% hold on
% [row, col] = find(train_data(:, 1));
% plot (centroids(row,1), centroids(row,2), 'y*');
% text(centroids(row,1), centroids(row,2), num2str(row),'horizontal','center', 'vertical','middle','BackgroundColor', [1 1 1]);     
% hold off
% 
% figure(8); imshow(im_lb);

%===========Machine Learning============
load('mobile_im1_data_2');
mobile_im1_data = train_data;
load('mobile_im2_data_2');
mobile_im2_data = train_data;
load('mobile_im3_data_2');
mobile_im3_data = train_data;
final_train_data = [mobile_im1_data; mobile_im2_data; mobile_im3_data];
%=============Cross Validation -- Bagging=============
bag_accu = zeros(1, 3);
true_bag_accu = zeros(1, 3);
bag_accu_per_point = zeros(1, 3);
true_bag_accu_per_point = zeros(1, 3);
cell_count = zeros(1, 3);
true_count = [594, 563, 737];
for i = 1 : 3
    if (i == 1)
        s_validate = 1;
        e_validate = length(mobile_im1_data);
    elseif (i == 2)
        s_validate = length(mobile_im1_data) + 1;
        e_validate = s_validate + length(mobile_im2_data) - 1;
    elseif (i == 3)
        s_validate = length(mobile_im1_data) + length(mobile_im2_data) + 1;
        e_validate = s_validate + length(mobile_im3_data) - 1;
    end
    validate_data = final_train_data(s_validate : e_validate, :);
    if (i == 1)
        train = final_train_data(length(mobile_im1_data) + 1 : end, :);
    elseif (i == 3)
        train = final_train_data(1 : length(final_train_data)-length(mobile_im3_data), :);
    else
        train = [final_train_data(1 : s_validate-1, :); final_train_data(e_validate+1 : end , :)];
    end
    % final_trainingdata includes labels at the last column
    ens_classifier = fitensemble(train(:, 2:end), train(:, 1), 'bag', 500, 'Tree', 'type', 'classification');
    ens_predictions = predict(ens_classifier, validate_data(:, 2:end));
    cell_count(i) = nnz(ens_predictions);
    bag_accu(i) = 1 - abs(nnz(ens_predictions) / nnz(validate_data(:, 1)) - 1);
    true_bag_accu(i) = 1 - abs(nnz(ens_predictions) / true_count(i) - 1);
    bag_accu_per_point(i) = 1 - nnz(ens_predictions - validate_data(:, 1)) / length(validate_data(:, 1));
    true_bag_accu_per_point(i) = 1 - nnz(ens_predictions - validate_data(:, 1)) / true_count(i);
end

bag_accu %#ok<*NOPTS>

true_bag_accu

bag_accu_per_point

true_bag_accu_per_point

identified_label_accuracy = num_identified_labels ./ true_count

cell_count