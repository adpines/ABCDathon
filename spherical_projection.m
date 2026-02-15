% Spherical projection of brain data
% Projects cortical BOLD timeseries onto a 2D equirectangular grid
% where x = azimuth and y = elevation from spherical surface coordinates
%
% Requires: FieldTrip (ft_read_cifti), GIfTI library
% Inputs:   Spherical surface .gii files (fs_LR 32k), CIFTI dtseries
% Outputs:  2D grids of brain activity per timepoint

addpath(genpath('E:\codelib'))

%% 1. SETUP - Set paths
sphere_L_path = 'E:/Downloads/fs_LR.32k.L.sphere.surf.gii';
sphere_R_path = 'E:/Downloads/fs_LR.32k.R.sphere.surf.gii';
dtseries_path = 'E:/Downloads/sub-s002_ses-V9_task-Resting1NewHB6scan_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii';

% Grid resolution in degrees
grid_step = 2;  % degrees per pixel

% Grid ranges (degrees)
az_range  = -180:grid_step:180;   % azimuth (x-axis)
el_range  = -90:grid_step:90;     % elevation (y-axis)

%% 2. LOAD DATA
fprintf('Loading data...\n');
data = ft_read_cifti(dtseries_path);
surf_L = gifti(sphere_L_path);
surf_R = gifti(sphere_R_path);

%% 3. PROCESS LEFT HEMISPHERE
fprintf('\nProcessing left hemisphere...\n');

% Convert Cartesian sphere vertices to spherical coordinates
[az_L, el_L, ~] = cart2sph(double(surf_L.vertices(:,1)), ...
                           double(surf_L.vertices(:,2)), ...
                           double(surf_L.vertices(:,3)));

% Convert from radians to degrees
az_L = rad2deg(az_L);  % azimuth:   -180 to 180
el_L = rad2deg(el_L);  % elevation:  -90 to  90

% Get cortical timeseries for left hemisphere
sig_L = data.dtseries(1:length(az_L), :)';  % [timepoints x vertices]

% Remove NaN vertices (medial wall)
validIdx_L = ~all(isnan(sig_L), 1);
az_L  = az_L(validIdx_L);
el_L  = el_L(validIdx_L);
sig_L = sig_L(:, validIdx_L);

fprintf('  Valid vertices: %d\n', length(az_L));

% Create regular grid
[azGrid_L, elGrid_L] = meshgrid(az_range, el_range);

% Build mask from convex hull of valid vertex positions
k_L = convhull(az_L, el_L);
mask_L = double(inpolygon(azGrid_L, elGrid_L, az_L(k_L), el_L(k_L)));
mask_L(mask_L == 0) = NaN;

fprintf('  Mask coverage: %.1f%%\n', 100*sum(~isnan(mask_L(:)))/numel(mask_L));

% Interpolate timeseries onto grid
nTimepoints = size(sig_L, 1);
data_grid_L = zeros(length(el_range), length(az_range), nTimepoints);

fprintf('  Interpolating %d timepoints...\n', nTimepoints);
for t = 1:nTimepoints
    interpData = griddata(az_L, el_L, sig_L(t,:)', azGrid_L, elGrid_L, 'linear');
    data_grid_L(:,:,t) = interpData .* mask_L;

    if mod(t, 50) == 0
        fprintf('    %d/%d\n', t, nTimepoints);
    end
end

%% 4. PROCESS RIGHT HEMISPHERE
fprintf('\nProcessing right hemisphere...\n');

% Convert Cartesian sphere vertices to spherical coordinates
[az_R, el_R, ~] = cart2sph(double(surf_R.vertices(:,1)), ...
                           double(surf_R.vertices(:,2)), ...
                           double(surf_R.vertices(:,3)));

az_R = rad2deg(az_R);
el_R = rad2deg(el_R);

% Get cortical timeseries for right hemisphere
nVerts_L = size(surf_L.vertices, 1);
sig_R = data.dtseries(nVerts_L+1:nVerts_L+length(az_R), :)';

% Remove NaN vertices (medial wall)
validIdx_R = ~all(isnan(sig_R), 1);
az_R  = az_R(validIdx_R);
el_R  = el_R(validIdx_R);
sig_R = sig_R(:, validIdx_R);

fprintf('  Valid vertices: %d\n', length(az_R));

% Create regular grid
[azGrid_R, elGrid_R] = meshgrid(az_range, el_range);

% Build mask
k_R = convhull(az_R, el_R);
mask_R = double(inpolygon(azGrid_R, elGrid_R, az_R(k_R), el_R(k_R)));
mask_R(mask_R == 0) = NaN;

fprintf('  Mask coverage: %.1f%%\n', 100*sum(~isnan(mask_R(:)))/numel(mask_R));

% Interpolate timeseries onto grid
data_grid_R = zeros(length(el_range), length(az_range), nTimepoints);

fprintf('  Interpolating %d timepoints...\n', nTimepoints);
for t = 1:nTimepoints
    interpData = griddata(az_R, el_R, sig_R(t,:)', azGrid_R, elGrid_R, 'linear');
    data_grid_R(:,:,t) = interpData .* mask_R;

    if mod(t, 50) == 0
        fprintf('    %d/%d\n', t, nTimepoints);
    end
end

%% 5. VISUALIZE
fprintf('\nPlotting...\n');

timepoint = 6;
clims = [-60, 60];

% Single timepoint - both hemispheres
figure('Position', [100, 100, 1400, 600]);

subplot(1,2,1);
imagesc(az_range, el_range, data_grid_L(:,:,timepoint));
colormap(jet(256));
caxis(clims);
colorbar;
axis xy equal tight;
xlabel('Azimuth (deg)');
ylabel('Elevation (deg)');
title(sprintf('Left Hemisphere - t=%d', timepoint));

subplot(1,2,2);
imagesc(az_range, el_range, data_grid_R(:,:,timepoint));
colormap(jet(256));
caxis(clims);
colorbar;
axis xy equal tight;
xlabel('Azimuth (deg)');
ylabel('Elevation (deg)');
title(sprintf('Right Hemisphere - t=%d', timepoint));

sgtitle('Spherical Projection (Equirectangular)', 'FontSize', 16);

% Masks
figure('Position', [100, 100, 1400, 600]);

subplot(1,2,1);
imagesc(az_range, el_range, mask_L);
axis xy equal tight;
xlabel('Azimuth (deg)');
ylabel('Elevation (deg)');
title('Left Mask');
colorbar;

subplot(1,2,2);
imagesc(az_range, el_range, mask_R);
axis xy equal tight;
xlabel('Azimuth (deg)');
ylabel('Elevation (deg)');
title('Right Mask');
colorbar;

sgtitle('Cortex Masks', 'FontSize', 16);

% Temporal mean
figure('Position', [100, 100, 1400, 600]);

mean_L = mean(data_grid_L, 3, 'omitnan');
mean_R = mean(data_grid_R, 3, 'omitnan');
clims_mean = [min([mean_L(:); mean_R(:)], [], 'omitnan'), ...
              max([mean_L(:); mean_R(:)], [], 'omitnan')];

subplot(1,2,1);
imagesc(az_range, el_range, mean_L);
colormap(hot(256));
caxis(clims_mean);
colorbar;
axis xy equal tight;
xlabel('Azimuth (deg)');
ylabel('Elevation (deg)');
title('Left - Temporal Mean');

subplot(1,2,2);
imagesc(az_range, el_range, mean_R);
colormap(hot(256));
caxis(clims_mean);
colorbar;
axis xy equal tight;
xlabel('Azimuth (deg)');
ylabel('Elevation (deg)');
title('Right - Temporal Mean');

sgtitle('Temporal Mean Activity', 'FontSize', 16);

%% 6. SAVE INDIVIDUAL HEMISPHERES
fprintf('\nSaving results...\n');
save('spherical_projection_results.mat', 'data_grid_L', 'data_grid_R', ...
     'mask_L', 'mask_R', 'az_range', 'el_range', 'nTimepoints', 'grid_step');

fprintf('  Left:  [%d x %d x %d]\n', size(data_grid_L));
fprintf('  Right: [%d x %d x %d]\n', size(data_grid_R));

%% 7. MERGE HEMISPHERES
fprintf('\nMerging hemispheres...\n');

data_grid_merged = cat(2, data_grid_L, data_grid_R);
mask_merged = cat(2, mask_L, mask_R);

fprintf('  Merged dimensions: [%d x %d x %d]\n', size(data_grid_merged));

% Merged x-axis: left hemisphere azimuth then right hemisphere azimuth
az_merged = [az_range, az_range + 360 + grid_step];

figure('Position', [100, 100, 1600, 500]);
imagesc(az_merged, el_range, data_grid_merged(:,:,timepoint));
colormap(jet(256));
caxis(clims);
colorbar;
axis xy equal tight;
xlabel('Azimuth (deg) — Left | Right');
ylabel('Elevation (deg)');
title(sprintf('Merged Hemispheres - t=%d', timepoint), 'FontSize', 16);

%% 8. SAVE MERGED DATA
fprintf('\nSaving merged data...\n');
save('spherical_projection_merged.mat', 'data_grid_merged', 'mask_merged', ...
     'az_range', 'el_range', 'az_merged', 'nTimepoints', 'grid_step');

fprintf('Done! Merged data saved.\n');
