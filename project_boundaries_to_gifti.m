% flatmap resampling + project grid boundaries back to gifti
%
% Part 1: Resample CIFTI timeseries onto regular 2D grid via flatmap
% Part 2: Map grid rectangle boundaries back to gifti surface vertices
%         so boundaries can be visualized in native brain space

addpath(genpath('E:\codelib'))

%% 1. SETUP - Set paths
flatmap_L_path = 'E:/Downloads/fs_LR.32k.L.flat.surf.gii';
flatmap_R_path = 'E:/Downloads/fs_LR.32k.R.flat.surf.gii';
dtseries_path = 'E:/Downloads/sub-s002_ses-V9_task-Resting1NewHB6scan_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii';

% Parameters
downsample = 2;
xCord_L = -250:downsample:250;
yCord_L = -150:downsample:200;
xCord_R = -270:downsample:230;
yCord_R = -180:downsample:170;

%% 2. LOAD DATA
fprintf('Loading data...\n');
data = ft_read_cifti(dtseries_path);
surf_L = gifti(flatmap_L_path);
surf_R = gifti(flatmap_R_path);

%% 3. PROCESS LEFT HEMISPHERE
fprintf('\nProcessing left hemisphere...\n');

% Extract positions
x_L = double(surf_L.vertices(:,1));
y_L = double(surf_L.vertices(:,2));

% Get cortical timeseries data for left hemisphere
sig_L = data.dtseries(1:length(x_L), :)';  % [timepoints x vertices]

% Remove NaN vertices (medial wall)
validIdx_L = ~all(isnan(sig_L), 1);
x_L = x_L(validIdx_L);
y_L = y_L(validIdx_L);
sig_L = sig_L(:, validIdx_L);

fprintf('  Valid vertices: %d\n', length(x_L));

% Create alpha shape boundary
k_L = alphaShape(x_L, y_L, 4);
[~, boundary_L] = k_L.boundaryFacets();

% Create mask from boundary
boundary_L_x_idx = (boundary_L(:,1) - min(xCord_L)) / downsample + 1;
boundary_L_y_idx = (boundary_L(:,2) - min(yCord_L)) / downsample + 1;

bw_L = poly2mask(boundary_L_x_idx, boundary_L_y_idx, ...
                 length(yCord_L), length(xCord_L));
mask_L = double(bw_L);
mask_L(mask_L == 0) = nan;

fprintf('  Mask coverage: %.1f%%\n', 100*sum(~isnan(mask_L(:)))/numel(mask_L));

% Interpolate data onto grid
[xGrid_L, yGrid_L] = meshgrid(xCord_L, yCord_L);
nTimepoints = size(sig_L, 1);
data_grid_L = zeros(length(yCord_L), length(xCord_L), nTimepoints);

fprintf('  Interpolating %d timepoints...\n', nTimepoints);
for t = 1:nTimepoints
    interpData = griddata(x_L, y_L, sig_L(t,:)', xGrid_L, yGrid_L, 'linear');
    data_grid_L(:,:,t) = interpData .* mask_L;

    if mod(t, 50) == 0
        fprintf('    %d/%d\n', t, nTimepoints);
    end
end

%% 4. PROCESS RIGHT HEMISPHERE
fprintf('\nProcessing right hemisphere...\n');

% Extract positions
x_R = double(surf_R.vertices(:,1));
y_R = double(surf_R.vertices(:,2));

% Get cortical timeseries data for right hemisphere
sig_R = data.dtseries(length(surf_L.vertices)+1:length(surf_L.vertices)+length(x_R), :)';

% Remove NaN vertices
validIdx_R = ~all(isnan(sig_R), 1);
x_R = x_R(validIdx_R);
y_R = y_R(validIdx_R);
sig_R = sig_R(:, validIdx_R);

fprintf('  Valid vertices: %d\n', length(x_R));

% Create alpha shape boundary
k_R = alphaShape(x_R, y_R, 4);
[~, boundary_R] = k_R.boundaryFacets();

% Create mask
boundary_R_x_idx = (boundary_R(:,1) - min(xCord_R)) / downsample + 1;
boundary_R_y_idx = (boundary_R(:,2) - min(yCord_R)) / downsample + 1;

bw_R = poly2mask(boundary_R_x_idx, boundary_R_y_idx, ...
                 length(yCord_R), length(xCord_R));
mask_R = double(bw_R);
mask_R(mask_R == 0) = nan;

fprintf('  Mask coverage: %.1f%%\n', 100*sum(~isnan(mask_R(:)))/numel(mask_R));

% Interpolate data onto grid
[xGrid_R, yGrid_R] = meshgrid(xCord_R, yCord_R);
data_grid_R = zeros(length(yCord_R), length(xCord_R), nTimepoints);

fprintf('  Interpolating %d timepoints...\n', nTimepoints);
for t = 1:nTimepoints
    interpData = griddata(x_R, y_R, sig_R(t,:)', xGrid_R, yGrid_R, 'linear');
    data_grid_R(:,:,t) = interpData .* mask_R;

    if mod(t, 50) == 0
        fprintf('    %d/%d\n', t, nTimepoints);
    end
end

%% 5. VISUALIZE
fprintf('\nPlotting...\n');

timepoint = 1;
figure('Position', [100, 100, 1400, 600]);

clims = [-60, 60];

subplot(1,2,1);
imagesc(data_grid_L(:,:,timepoint));
colormap(jet(256));
caxis(clims);
colorbar;
axis off; axis equal tight;
title(sprintf('Left Hemisphere - t=%d', timepoint));

subplot(1,2,2);
imagesc(data_grid_R(:,:,timepoint));
colormap(jet(256));
caxis(clims);
colorbar;
axis off; axis equal tight;
title(sprintf('Right Hemisphere - t=%d', timepoint));

sgtitle('Flatmap Projection', 'FontSize', 16);

% Plot masks
figure('Position', [100, 100, 1400, 600]);
subplot(1,2,1);
imagesc(mask_L);
title('Left Mask');
colorbar;
axis off; axis equal tight;

subplot(1,2,2);
imagesc(mask_R);
title('Right Mask');
colorbar;
axis off; axis equal tight;

sgtitle('Cortex Masks', 'FontSize', 16);

% Plot temporal mean
figure('Position', [100, 100, 1400, 600]);

mean_L = mean(data_grid_L, 3, 'omitnan');
mean_R = mean(data_grid_R, 3, 'omitnan');
clims_mean = [min([mean_L(:); mean_R(:)], [], 'omitnan'), max([mean_L(:); mean_R(:)], [], 'omitnan')];

subplot(1,2,1);
imagesc(mean_L);
colormap(hot(256));
caxis(clims_mean);
colorbar;
axis off; axis equal tight;
title('Left - Temporal Mean');

subplot(1,2,2);
imagesc(mean_R);
colormap(hot(256));
caxis(clims_mean);
colorbar;
axis off; axis equal tight;
title('Right - Temporal Mean');

sgtitle('Temporal Mean Activity', 'FontSize', 16);

%% 6. SAVE GRID DATA
fprintf('\nSaving results...\n');
save('flatmap_projection_results.mat', 'data_grid_L', 'data_grid_R', ...
     'mask_L', 'mask_R', 'xCord_L', 'yCord_L', 'xCord_R', 'yCord_R', ...
     'nTimepoints');

fprintf('\nDone!\n');
fprintf('Output dimensions:\n');
fprintf('  Left:  [%d x %d x %d]\n', size(data_grid_L));
fprintf('  Right: [%d x %d x %d]\n', size(data_grid_R));


%% 7. MAP GRID BOUNDARIES BACK TO GIFTI SURFACE
% The grid boundaries are the edges of the resampling rectangle in flatmap
% 2D space. The flatmap vertices live in the same 2D coordinate system.
% So we just need to find which flatmap vertices are near the edges of
% the grid rectangle, then write those labels out as a func.gii that can
% be loaded onto the native (inflated/pial) surface for visualization.
%
% Boundary labels:
%   0 = not on any boundary
%   1 = left edge   (x = min of xCord)
%   2 = right edge  (x = max of xCord)
%   3 = bottom edge (y = min of yCord)
%   4 = top edge    (y = max of yCord)

fprintf('\nMapping grid boundaries back to surface...\n');

% Distance threshold in flatmap units. The grid spacing is `downsample`
% units, so marking vertices within one grid cell of the boundary edge
% is a reasonable choice. Adjust if lines are too thick or too thin.
edge_threshold = downsample;

%% LEFT HEMISPHERE
fprintf('Processing left hemisphere boundaries...\n');

% Get ALL flatmap vertex positions (including medial wall)
x_L_all = double(surf_L.vertices(:,1));
y_L_all = double(surf_L.vertices(:,2));
nVerts_L = length(x_L_all);

% Grid rectangle edges for the left hemisphere
xmin_L = min(xCord_L);
xmax_L = max(xCord_L);
ymin_L = min(yCord_L);
ymax_L = max(yCord_L);

% Initialize boundary labels (0 = no boundary)
boundary_labels_L = zeros(nVerts_L, 1);

% Only label vertices that are inside the grid rectangle (plus threshold),
% so we don't mark distant medial wall vertices that happen to share an
% x or y coordinate with a boundary edge.
inside_x_L = (x_L_all >= xmin_L - edge_threshold) & (x_L_all <= xmax_L + edge_threshold);
inside_y_L = (y_L_all >= ymin_L - edge_threshold) & (y_L_all <= ymax_L + edge_threshold);
inside_rect_L = inside_x_L & inside_y_L;

% Left edge: vertices near x = xmin, within the y range of the grid
near_left_L  = abs(x_L_all - xmin_L) < edge_threshold & inside_y_L;
% Right edge: vertices near x = xmax, within the y range
near_right_L = abs(x_L_all - xmax_L) < edge_threshold & inside_y_L;
% Bottom edge: vertices near y = ymin, within the x range
near_bot_L   = abs(y_L_all - ymin_L) < edge_threshold & inside_x_L;
% Top edge: vertices near y = ymax, within the x range
near_top_L   = abs(y_L_all - ymax_L) < edge_threshold & inside_x_L;

% Assign labels (later labels don't overwrite earlier ones at corners)
boundary_labels_L(near_left_L)  = 1;
boundary_labels_L(near_right_L & boundary_labels_L == 0) = 2;
boundary_labels_L(near_bot_L   & boundary_labels_L == 0) = 3;
boundary_labels_L(near_top_L   & boundary_labels_L == 0) = 4;

fprintf('  Grid rectangle: x=[%.0f, %.0f], y=[%.0f, %.0f]\n', ...
        xmin_L, xmax_L, ymin_L, ymax_L);
fprintf('  Edge threshold: %.1f flatmap units\n', edge_threshold);
fprintf('  Boundary vertex counts: left=%d, right=%d, bottom=%d, top=%d\n', ...
        sum(boundary_labels_L==1), sum(boundary_labels_L==2), ...
        sum(boundary_labels_L==3), sum(boundary_labels_L==4));

%% RIGHT HEMISPHERE
fprintf('Processing right hemisphere boundaries...\n');

x_R_all = double(surf_R.vertices(:,1));
y_R_all = double(surf_R.vertices(:,2));
nVerts_R = length(x_R_all);

xmin_R = min(xCord_R);
xmax_R = max(xCord_R);
ymin_R = min(yCord_R);
ymax_R = max(yCord_R);

boundary_labels_R = zeros(nVerts_R, 1);

inside_x_R = (x_R_all >= xmin_R - edge_threshold) & (x_R_all <= xmax_R + edge_threshold);
inside_y_R = (y_R_all >= ymin_R - edge_threshold) & (y_R_all <= ymax_R + edge_threshold);

near_left_R  = abs(x_R_all - xmin_R) < edge_threshold & inside_y_R;
near_right_R = abs(x_R_all - xmax_R) < edge_threshold & inside_y_R;
near_bot_R   = abs(y_R_all - ymin_R) < edge_threshold & inside_x_R;
near_top_R   = abs(y_R_all - ymax_R) < edge_threshold & inside_x_R;

boundary_labels_R(near_left_R)  = 1;
boundary_labels_R(near_right_R & boundary_labels_R == 0) = 2;
boundary_labels_R(near_bot_R   & boundary_labels_R == 0) = 3;
boundary_labels_R(near_top_R   & boundary_labels_R == 0) = 4;

fprintf('  Grid rectangle: x=[%.0f, %.0f], y=[%.0f, %.0f]\n', ...
        xmin_R, xmax_R, ymin_R, ymax_R);
fprintf('  Boundary vertex counts: left=%d, right=%d, bottom=%d, top=%d\n', ...
        sum(boundary_labels_R==1), sum(boundary_labels_R==2), ...
        sum(boundary_labels_R==3), sum(boundary_labels_R==4));

%% SAVE BOUNDARY GIFTI FILES
fprintf('\nSaving boundary gifti files...\n');

g_L = gifti();
g_L.cdata = single(boundary_labels_L);
save(g_L, 'E:\Downloads\left_hemisphere_grid_boundaries.func.gii');

g_R = gifti();
g_R.cdata = single(boundary_labels_R);
save(g_R, 'E:\Downloads\right_hemisphere_grid_boundaries.func.gii');

fprintf('Saved:\n');
fprintf('  E:\\Downloads\\left_hemisphere_grid_boundaries.func.gii\n');
fprintf('  E:\\Downloads\\right_hemisphere_grid_boundaries.func.gii\n');
fprintf('\nBoundary labels:\n');
fprintf('  0 = not on boundary\n');
fprintf('  1 = left edge   (x = %d)\n', xmin_L);
fprintf('  2 = right edge  (x = %d)\n', xmax_L);
fprintf('  3 = bottom edge (y = %d)\n', ymin_L);
fprintf('  4 = top edge    (y = %d)\n', ymax_L);
fprintf('\nVisualize by loading .func.gii onto inflated or pial surface in wb_view.\n');
