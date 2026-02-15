% Spherical projection with medial wall rotation
% Rotates medial wall to pole, then maps to azimuth/elevation grid
% Part 2 maps grid boundaries back to gifti for visualization in wb_view

addpath(genpath('E:\codelib'))

%% 1. SETUP
sphere_L_path = 'E:/Downloads/fs_LR.32k.L.sphere.surf.gii';
sphere_R_path = 'E:/Downloads/fs_LR.32k.R.sphere.surf.gii';
dtseries_path = 'E:/Downloads/sub-s002_ses-V9_task-Resting1NewHB6scan_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii';

% Grid parameters
downsample = 4;
azimuth_range = -180:downsample:180;
elevation_range = -90:downsample:90;

%% 2. LOAD DATA
fprintf('Loading data...\n');
data = ft_read_cifti(dtseries_path);
surf_L = gifti(sphere_L_path);
surf_R = gifti(sphere_R_path);

%% 3. PROCESS LEFT HEMISPHERE
fprintf('\nProcessing left hemisphere...\n');

% Get xyz coordinates and timeseries
x_L = double(surf_L.vertices(:,1));
y_L = double(surf_L.vertices(:,2));
z_L = double(surf_L.vertices(:,3));
sig_L = data.dtseries(1:length(x_L), :)';  % [timepoints x vertices]

% Identify medial wall (NaN vertices)
medialIdx_L = all(isnan(sig_L), 1);
fprintf('  Medial wall vertices: %d\n', sum(medialIdx_L));

% Find mean direction of medial wall
medial_dir_L = [mean(x_L(medialIdx_L)); mean(y_L(medialIdx_L)); mean(z_L(medialIdx_L))];
medial_dir_L = medial_dir_L / norm(medial_dir_L);
fprintf('  Medial wall direction: [%.3f, %.3f, %.3f]\n', medial_dir_L);

% Calculate rotation to move medial wall to north pole [0, 0, 1]
north_pole = [0; 0; 1];
rotation_axis_L = cross(medial_dir_L, north_pole);
rotation_axis_L = rotation_axis_L / norm(rotation_axis_L);
rotation_angle_L = acos(dot(medial_dir_L, north_pole));

% Build rotation matrix using Rodrigues formula
K_L = [0, -rotation_axis_L(3), rotation_axis_L(2);
       rotation_axis_L(3), 0, -rotation_axis_L(1);
       -rotation_axis_L(2), rotation_axis_L(1), 0];
R_L = eye(3) + sin(rotation_angle_L)*K_L + (1-cos(rotation_angle_L))*(K_L*K_L);

% Apply rotation to all vertices
vertices_rotated_L = R_L * [x_L, y_L, z_L]';  % [3 x N]
x_L_rot = vertices_rotated_L(1,:)';
y_L_rot = vertices_rotated_L(2,:)';
z_L_rot = vertices_rotated_L(3,:)';

% Keep only valid (non-medial) vertices for interpolation
validIdx_L = ~medialIdx_L;
x_L_valid = x_L_rot(validIdx_L);
y_L_valid = y_L_rot(validIdx_L);
z_L_valid = z_L_rot(validIdx_L);
sig_L_valid = sig_L(:, validIdx_L);
fprintf('  Valid vertices: %d\n', sum(validIdx_L));

% Convert rotated xyz to azimuth/elevation
azimuth_L = atan2d(y_L_valid, x_L_valid);
elevation_L = atan2d(z_L_valid, sqrt(x_L_valid.^2 + y_L_valid.^2));

% Interpolate onto grid
[azGrid_L, elGrid_L] = meshgrid(azimuth_range, elevation_range);
nTimepoints = size(sig_L, 1);
data_grid_L = zeros(length(elevation_range), length(azimuth_range), nTimepoints);

fprintf('  Interpolating %d timepoints...\n', nTimepoints);
for t = 1:nTimepoints
    F = scatteredInterpolant(azimuth_L, elevation_L, sig_L_valid(t,:)', 'linear', 'none');
    data_grid_L(:,:,t) = F(azGrid_L, elGrid_L);
    if mod(t, 50) == 0
        fprintf('    %d/%d\n', t, nTimepoints);
    end
end

%% 4. PROCESS RIGHT HEMISPHERE
fprintf('\nProcessing right hemisphere...\n');

x_R = double(surf_R.vertices(:,1));
y_R = double(surf_R.vertices(:,2));
z_R = double(surf_R.vertices(:,3));
sig_R = data.dtseries(length(surf_L.vertices)+1:length(surf_L.vertices)+length(x_R), :)';

% Identify medial wall
medialIdx_R = all(isnan(sig_R), 1);
fprintf('  Medial wall vertices: %d\n', sum(medialIdx_R));

% Find mean direction and rotate to north pole
medial_dir_R = [mean(x_R(medialIdx_R)); mean(y_R(medialIdx_R)); mean(z_R(medialIdx_R))];
medial_dir_R = medial_dir_R / norm(medial_dir_R);
fprintf('  Medial wall direction: [%.3f, %.3f, %.3f]\n', medial_dir_R);

rotation_axis_R = cross(medial_dir_R, north_pole);
rotation_axis_R = rotation_axis_R / norm(rotation_axis_R);
rotation_angle_R = acos(dot(medial_dir_R, north_pole));

K_R = [0, -rotation_axis_R(3), rotation_axis_R(2);
       rotation_axis_R(3), 0, -rotation_axis_R(1);
       -rotation_axis_R(2), rotation_axis_R(1), 0];
R_R = eye(3) + sin(rotation_angle_R)*K_R + (1-cos(rotation_angle_R))*(K_R*K_R);

vertices_rotated_R = R_R * [x_R, y_R, z_R]';
x_R_rot = vertices_rotated_R(1,:)';
y_R_rot = vertices_rotated_R(2,:)';
z_R_rot = vertices_rotated_R(3,:)';

validIdx_R = ~medialIdx_R;
x_R_valid = x_R_rot(validIdx_R);
y_R_valid = y_R_rot(validIdx_R);
z_R_valid = z_R_rot(validIdx_R);
sig_R_valid = sig_R(:, validIdx_R);
fprintf('  Valid vertices: %d\n', sum(validIdx_R));

azimuth_R = atan2d(y_R_valid, x_R_valid);
elevation_R = atan2d(z_R_valid, sqrt(x_R_valid.^2 + y_R_valid.^2));

[azGrid_R, elGrid_R] = meshgrid(azimuth_range, elevation_range);
data_grid_R = zeros(length(elevation_range), length(azimuth_range), nTimepoints);

fprintf('  Interpolating %d timepoints...\n', nTimepoints);
for t = 1:nTimepoints
    F = scatteredInterpolant(azimuth_R, elevation_R, sig_R_valid(t,:)', 'linear', 'none');
    data_grid_R(:,:,t) = F(azGrid_R, elGrid_R);
    if mod(t, 50) == 0
        fprintf('    %d/%d\n', t, nTimepoints);
    end
end

%% 5. VISUALIZE
fprintf('\nPlotting...\n');

timepoint = 6;
clims = [-60, 60];

figure('Position', [100, 100, 1400, 600]);
subplot(1,2,1);
imagesc(data_grid_L(:,:,timepoint));
colormap(jet(256));
caxis(clims);
colorbar;
xlabel('Azimuth index');
ylabel('Elevation index');
title(sprintf('Left Hemisphere - t=%d', timepoint));

subplot(1,2,2);
imagesc(data_grid_R(:,:,timepoint));
colormap(jet(256));
caxis(clims);
colorbar;
xlabel('Azimuth index');
ylabel('Elevation index');
title(sprintf('Right Hemisphere - t=%d', timepoint));

sgtitle('Spherical Projection (Medial Wall at Pole)', 'FontSize', 16);

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
xlabel('Azimuth index');
ylabel('Elevation index');
title('Left - Temporal Mean');

subplot(1,2,2);
imagesc(mean_R);
colormap(hot(256));
caxis(clims_mean);
colorbar;
xlabel('Azimuth index');
ylabel('Elevation index');
title('Right - Temporal Mean');

sgtitle('Temporal Mean Activity', 'FontSize', 16);

%% 6. SAVE
fprintf('\nSaving results...\n');
save('spherical_projection_rotated.mat', 'data_grid_L', 'data_grid_R', ...
     'azimuth_range', 'elevation_range', 'nTimepoints');

fprintf('\nDone!\n');
fprintf('Output dimensions:\n');
fprintf('  Left:  [%d x %d x %d]\n', size(data_grid_L));
fprintf('  Right: [%d x %d x %d]\n', size(data_grid_R));


%% 7. MAP GRID BOUNDARIES BACK TO GIFTI SURFACE
%
% The grid edges in the image correspond to lines of constant azimuth or
% elevation in the rotated spherical frame. To project these back to the
% original sphere, we compute azimuth/elevation for ALL vertices in the
% rotated frame and mark those near the grid edges.
%
% Key geometry facts for equirectangular sphere projection:
%   - az=-180 and az=+180 are the SAME meridian (azimuth wraps around).
%     So left/right image edges map to one line on the brain.
%   - el=+90 is the north pole (medial wall after rotation).
%   - el=-90 is the south pole (a single point, not a line).
%     Near the south pole, ALL azimuths converge, so "bottom edge"
%     is a small cap rather than a line.
%
% Boundary labels (bitmask for overlap at corners/intersections):
%   0 = not on any boundary
%   1 = azimuth seam (az near +/-180, i.e. left/right image edges)
%   2 = south pole cap (el near -90, i.e. bottom image edge)
%   4 = north pole cap / medial wall border (el near +90, i.e. top edge)
%
% To visualize: load the .func.gii onto the inflated or pial surface
% (same 32k mesh) in wb_view. Threshold to show values > 0.

fprintf('\nMapping grid boundaries back to surface...\n');

% Threshold in degrees. Controls how thick the boundary lines appear.
% Larger = thicker lines, more vertices marked. Start with downsample
% value (one grid cell width in degrees).
edge_threshold_deg = downsample;

%% LEFT HEMISPHERE BOUNDARIES
fprintf('Processing left hemisphere boundaries...\n');

% Compute azimuth/elevation for ALL left vertices in the rotated frame
% (R_L and rotated coordinates already computed in Part 1)
az_all_L = atan2d(y_L_rot, x_L_rot);
el_all_L = atan2d(z_L_rot, sqrt(x_L_rot.^2 + y_L_rot.^2));

boundary_labels_L = zeros(length(x_L), 1);

% Azimuth seam: |az| near 180 — the single meridian where the image wraps.
% This is where the left and right edges of your grid image meet on the
% sphere. Use wrapped distance: min(|az - 180|, |az + 180|) = 180 - |az|.
near_az_seam_L = (180 - abs(az_all_L)) < edge_threshold_deg;
boundary_labels_L(near_az_seam_L) = 1;

% South pole cap: el near -90.
near_south_L = (el_all_L - (-90)) < edge_threshold_deg;
boundary_labels_L(near_south_L) = boundary_labels_L(near_south_L) + 2;

% North pole / medial wall border: el near +90.
near_north_L = (90 - el_all_L) < edge_threshold_deg;
boundary_labels_L(near_north_L) = boundary_labels_L(near_north_L) + 4;

fprintf('  Edge threshold: %.1f degrees\n', edge_threshold_deg);
fprintf('  Boundary counts: az_seam=%d, south_pole=%d, north_pole/medial=%d\n', ...
        sum(near_az_seam_L), sum(near_south_L), sum(near_north_L));

%% RIGHT HEMISPHERE BOUNDARIES
fprintf('Processing right hemisphere boundaries...\n');

az_all_R = atan2d(y_R_rot, x_R_rot);
el_all_R = atan2d(z_R_rot, sqrt(x_R_rot.^2 + y_R_rot.^2));

boundary_labels_R = zeros(length(x_R), 1);

near_az_seam_R = (180 - abs(az_all_R)) < edge_threshold_deg;
boundary_labels_R(near_az_seam_R) = 1;

near_south_R = (el_all_R - (-90)) < edge_threshold_deg;
boundary_labels_R(near_south_R) = boundary_labels_R(near_south_R) + 2;

near_north_R = (90 - el_all_R) < edge_threshold_deg;
boundary_labels_R(near_north_R) = boundary_labels_R(near_north_R) + 4;

fprintf('  Boundary counts: az_seam=%d, south_pole=%d, north_pole/medial=%d\n', ...
        sum(near_az_seam_R), sum(near_south_R), sum(near_north_R));

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
fprintf('\nBoundary labels (bitmask):\n');
fprintf('  0 = not on boundary\n');
fprintf('  1 = azimuth seam (left/right image edges, az = +/-180)\n');
fprintf('  2 = south pole cap (bottom image edge, el = -90)\n');
fprintf('  4 = north pole / medial wall border (top image edge, el = +90)\n');
fprintf('  3 = azimuth seam + south pole overlap\n');
fprintf('  5 = azimuth seam + north pole overlap\n');
fprintf('\nVisualize: load .func.gii onto inflated/pial surface in wb_view.\n');
fprintf('Adjust edge_threshold_deg (currently %.1f) for thicker/thinner lines.\n', edge_threshold_deg);
