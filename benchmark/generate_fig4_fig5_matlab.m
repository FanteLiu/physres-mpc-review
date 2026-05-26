% Generate paper-ready Fig. 4 and Fig. 5 from the new benchmark dataset.
% Style target: clean Nature/Science-like time-series plots, Arial 8 pt.

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
releaseRoot = fullfile(scriptDir, '..');
dataFile = fullfile(scriptDir, 'paper_prediction_data.csv');
outDir = fullfile(releaseRoot, 'figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

T = readtable(dataFile);

warmup = 10;
idx = (warmup + 1):height(T);
time = T.time_s(idx) - T.time_s(idx(1));

% Colorblind-friendly, muted palette.
colors.actual = [0.00, 0.00, 0.00];
colors.physres = [0.80, 0.10, 0.10];
colors.linear = [0.45, 0.45, 0.45];
colors.mlp = [0.00, 0.45, 0.70];
colors.esn = [0.49, 0.18, 0.56];
colors.lstm = [0.00, 0.60, 0.50];

paperFont = 'Arial';
fontSize = 8;
legendFontSize = 6.8;

%% Fig. 4: Pressure prediction comparison
fig4 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2, 2, 8.9, 6.4]);
ax = axes(fig4);
hold(ax, 'on');

plot(ax, time, T.P_actual_kPa(idx), '-', 'Color', colors.actual, 'LineWidth', 1.55, ...
    'DisplayName', 'Actual');
plot(ax, time, T.P_PhysRes_kPa(idx), '-', 'Color', colors.physres, 'LineWidth', 1.45, ...
    'DisplayName', 'PhysRes');
plot(ax, time, T.P_Linear_AR_kPa(idx), ':', 'Color', colors.linear, 'LineWidth', 1.15, ...
    'DisplayName', 'Linear AR');
plot(ax, time, T.P_MLP_kPa(idx), '--', 'Color', colors.mlp, 'LineWidth', 1.15, ...
    'DisplayName', 'MLP');
plot(ax, time, T.P_ESN_kPa(idx), '-.', 'Color', colors.esn, 'LineWidth', 1.15, ...
    'DisplayName', 'ESN');
plot(ax, time, T.P_LSTM_kPa(idx), '--', 'Color', colors.lstm, 'LineWidth', 1.15, ...
    'DisplayName', 'LSTM');

xlabel(ax, 'Time (s)');
ylabel(ax, 'Pressure (kPa)');
xlim(ax, [time(1), time(end)]);
ylim(ax, paddedLimits([T.P_actual_kPa(idx); T.P_MLP_kPa(idx); T.P_ESN_kPa(idx); T.P_LSTM_kPa(idx)]));
styleAxes(ax, paperFont, fontSize);
legend(ax, 'Location', 'northoutside', 'NumColumns', 3, 'Box', 'off', ...
    'FontName', paperFont, 'FontSize', legendFontSize);
annotation(fig4, 'textbox', [0.24, 0.27, 0.22, 0.07], ...
    'String', sprintf('Actual + PhysRes + LSTM\noverlap'), ...
    'FontName', paperFont, 'FontSize', 5.9, 'Color', [0.12, 0.12, 0.12], ...
    'BackgroundColor', [1, 1, 1], 'EdgeColor', [0.70, 0.70, 0.70], ...
    'LineWidth', 0.5, 'Margin', 3, 'FitBoxToText', 'on');

exportgraphics(fig4, fullfile(outDir, 'fig4_pressure_prediction_new.png'), 'Resolution', 600);
exportgraphics(fig4, fullfile(outDir, 'fig4_pressure_prediction_new.pdf'), 'ContentType', 'vector');
close(fig4);

%% Fig. 5: Temperature prediction comparison
fig5 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2, 2, 8.9, 8.4]);
tl = tiledlayout(fig5, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl, 1);
hold(ax1, 'on');
plot(ax1, time, T.Tp_actual_C(idx), '-', 'Color', colors.actual, 'LineWidth', 1.55, ...
    'DisplayName', 'Actual');
plot(ax1, time, T.Tp_PhysRes_C(idx), '-', 'Color', colors.physres, 'LineWidth', 1.45, ...
    'DisplayName', 'PhysRes');
plot(ax1, time, T.Tp_MLP_C(idx), '--', 'Color', colors.mlp, 'LineWidth', 1.15, ...
    'DisplayName', 'MLP');
plot(ax1, time, T.Tp_ESN_C(idx), '-.', 'Color', colors.esn, 'LineWidth', 1.15, ...
    'DisplayName', 'ESN');
plot(ax1, time, T.Tp_LSTM_C(idx), '--', 'Color', colors.lstm, 'LineWidth', 1.15, ...
    'DisplayName', 'LSTM');
ylabel(ax1, 'T_Peltier (deg C)');
xlim(ax1, [time(1), time(end)]);
ylim(ax1, paddedLimits([T.Tp_actual_C(idx); T.Tp_PhysRes_C(idx); T.Tp_MLP_C(idx); T.Tp_ESN_C(idx); T.Tp_LSTM_C(idx)]));
styleAxes(ax1, paperFont, fontSize);
text(ax1, 0.015, 0.90, 'a', 'Units', 'normalized', 'FontName', paperFont, ...
    'FontSize', fontSize, 'FontWeight', 'bold');
text(ax1, 0.37, 1.11, 'Trace overlap indicates small temperature errors', ...
    'Units', 'normalized', 'FontName', paperFont, 'FontSize', 5.7, ...
    'Color', [0.12, 0.12, 0.12], 'VerticalAlignment', 'top', ...
    'BackgroundColor', [1, 1, 1], 'EdgeColor', [0.70, 0.70, 0.70], ...
    'Margin', 2, 'Clipping', 'off');

ax2 = nexttile(tl, 2);
hold(ax2, 'on');
plot(ax2, time, T.Tl_actual_C(idx), '-', 'Color', colors.actual, 'LineWidth', 1.55, ...
    'DisplayName', 'Actual');
plot(ax2, time, T.Tl_PhysRes_C(idx), '-', 'Color', colors.physres, 'LineWidth', 1.45, ...
    'DisplayName', 'PhysRes');
plot(ax2, time, T.Tl_MLP_C(idx), '--', 'Color', colors.mlp, 'LineWidth', 1.15, ...
    'DisplayName', 'MLP');
plot(ax2, time, T.Tl_ESN_C(idx), '-.', 'Color', colors.esn, 'LineWidth', 1.15, ...
    'DisplayName', 'ESN');
plot(ax2, time, T.Tl_LSTM_C(idx), '--', 'Color', colors.lstm, 'LineWidth', 1.15, ...
    'DisplayName', 'LSTM');
xlabel(ax2, 'Time (s)');
ylabel(ax2, 'T_liquid (deg C)');
xlim(ax2, [time(1), time(end)]);
ylim(ax2, paddedLimits([T.Tl_actual_C(idx); T.Tl_PhysRes_C(idx); T.Tl_MLP_C(idx); T.Tl_ESN_C(idx); T.Tl_LSTM_C(idx)]));
styleAxes(ax2, paperFont, fontSize);
text(ax2, 0.015, 0.90, 'b', 'Units', 'normalized', 'FontName', paperFont, ...
    'FontSize', fontSize, 'FontWeight', 'bold');

lgd = legend(ax1, 'Location', 'northoutside', 'NumColumns', 3, 'Box', 'off', ...
    'FontName', paperFont, 'FontSize', legendFontSize);
lgd.Layout.Tile = 'north';

exportgraphics(fig5, fullfile(outDir, 'fig5_temperature_prediction_new.png'), 'Resolution', 600);
exportgraphics(fig5, fullfile(outDir, 'fig5_temperature_prediction_new.pdf'), 'ContentType', 'vector');
close(fig5);

fprintf('Saved figures to:\n');
fprintf('  %s\n', fullfile(outDir, 'fig4_pressure_prediction_new.png'));
fprintf('  %s\n', fullfile(outDir, 'fig5_temperature_prediction_new.png'));

function styleAxes(ax, fontName, fontSize)
    set(ax, 'FontName', fontName, 'FontSize', fontSize, ...
        'LineWidth', 0.75, 'TickDir', 'out', 'Box', 'off', ...
        'Layer', 'top', 'XColor', [0 0 0], 'YColor', [0 0 0]);
    ax.TickLength = [0.018, 0.018];
    grid(ax, 'off');
end

function lim = paddedLimits(values)
    values = values(isfinite(values));
    vmin = min(values);
    vmax = max(values);
    pad = 0.06 * (vmax - vmin);
    if pad <= 0
        pad = 1;
    end
    lim = [vmin - pad, vmax + pad];
end
