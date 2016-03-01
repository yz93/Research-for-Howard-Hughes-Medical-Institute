clear;
% =============================Load train data============================
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
train_data = [im1_data; im2_data; im3_data; im4_data; im5_data; im6_data];
% =========================Train Model====================================
ens_classifier = fitensemble(train_data(:, 2:end), train_data(:, 1), 'bag', 500, 'Tree', 'type', 'classification');
% =========================Generate Test Data and Test====================
number_of_cells = zeros(1, 34);
% The first 17 are 1000-to-1 images; the last 17 are 100-to-1 images.
hardcoded_names = {'Series006_1000' 'Series012_1000' 'Series015_1000' ...
    'Series018_1000' 'Series021_1000' 'Series024_1000' 'Series027_1000' ...
    'Series030_1000' 'Series036_1000' 'Series039_1000' 'Series042_1000' ...
    'Series045_1000' 'Series048_1000' 'Series053_1000' 'Series056_1000' ...
    'Series062_1000' 'Series065_1000' ...
    'Series025_z0_ch02' 'Series032_z0_ch02' 'Series036_z0_ch02' ...
    'Series040_z0_ch02' 'Series043_z0_ch02' 'Series049_z0_ch02' ...
    'Series052_z0_ch02' 'Series058_z0_ch02' 'Series061_z0_ch02' ...
    'Series064_z0_ch02' 'Series067_z0_ch02' 'Series070_z0_ch02' ...
    'Series073_z0_ch02' 'Series076_z0_ch02' 'Series079_z0_ch02' ...
    'Series084_z0_ch02' 'Series092_z0_ch02'};
for j = 1 : 35
    image_path = ['Z:\Winter 2016\HHMI\20160107RCP_Microscope\test_data_mapping\' hardcoded_names{j} '.tif'];
    save_test_data_name = ['test_' hardcoded_names{j}];
    save_results_name = ['predictions_' hardcoded_names{j}];
    im = imread(image_path);
    level = graythresh(im);
    level = 0.5*level;  % 0.6, 0.7
    bw_im = im2bw(im, level);

    points = regionprops(bw_im, im, 'Area','Centroid','Eccentricity','MajorAxisLength','MinorAxisLength','Perimeter','Solidity','PixelIdxList','PixelList','MaxIntensity','MinIntensity','WeightedCentroid','BoundingBox','MeanIntensity','Orientation','Extrema','Extent','ConvexHull');%,'MeanIntensity','Orientation'
    test_data = zeros(length(points), 16);
    centroids = cat (1, points.Centroid);
    for i = 1 : length(points) % write out the list of possible objects and their properties
        weighted_centroids = cat(1, points.WeightedCentroid);
        test_data (i,2) = double(points(i).Area);
        test_data (i,3) = double(centroids(i,1));
        test_data (i,4) = double(centroids(i,2));
        test_data (i,5) = double(weighted_centroids(i,1));
        test_data (i,6) = double(weighted_centroids(i,2)); 
        test_data(i,7) = size(points(i).ConvexHull, 1); % number of vertex of the smallest convex polygon that can contain the region
        test_data (i,8) = double(points(i).Area)/size(points(i).ConvexHull,1);%
        test_data (i,9) = double(points(i).Solidity); % Area/ConvexArea
        test_data (i,10)= double(points(i).Extent); % Area divided by the area of the bounding box
        test_data (i,11) = double(points(i).MeanIntensity);
        test_data (i,12) = double(points(i).MaxIntensity);
        test_data (i,13) = double(points(i).MinIntensity);
        test_data (i,14) = sqrt((test_data(i,3)-test_data(i,5))^2 + (test_data(i,4)-test_data(i,6))^2);
        test_data (i,15) = double(points(i).Perimeter)/double(points(i).Area); %Perimeter/Area;
        test_data (i,16) = double(points(i).MaxIntensity)-double(points(i).MinIntensity);%MaxIntensity subtract MinIntensity
    end
    test_data = test_data(:, 2:16);
    ens_predictions = predict(ens_classifier, test_data);
    % save the number of cells (aka, number of non-zero elements)
    number_of_cells(j) = nnz(ens_predictions);
    ens_predictions = [ens_predictions; nnz(ens_predictions)];
    save(save_test_data_name, 'test_data');
    save(save_results_name, 'ens_predictions');
end