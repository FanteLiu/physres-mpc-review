% Generate paper-ready Fig. 7: benchmark visualization.
% Style target: clean Nature/Science-like double-column figure, Arial 8 pt.

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
releaseRoot = fullfile(scriptDir, '..');
summaryFile = fullfile(scriptDir, 'benchmark_summary.csv');
outDir = fullfile(releaseRoot, 'figures');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

T = readtable(summaryFile);

models = {'PhysRes', 'LSTM', 'ESN', 'MLP'};
markerNames = {'p', 'o', 's', '^'};
markerSizes = [9.0, 6.2, 6.2, 6.2];
colors = [
    0.80, 0.10, 0.10;  % PhysRes
    0.00, 0.60, 0.50;  % LSTM
    0.49, 0.18, 0.56;  % ESN
    0.00, 0.45, 0.70   % MLP
];

paperFont = 'Arial';
fontSize = 8;
legendFontSize = 7.2;

n = numel(models);
worstR2 = zeros(n, 1);
ips = zeros(n, 1);
params = zeros(n, 1);

for i = 1:n
    rows = strcmp(T.Model, models{i});
    r2Vals = T.R2(rows);
    worstR2(i) = min(r2Vals(isfinite(r2Vals)));
    inferMs = T.Infer_time_ms(find(rows, 1, 'first'));
    ips(i) = 1000 / inferMs;
    params(i) = T.Params(find(rows, 1, 'first'));
end

fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2, 2, 17.8, 5.8]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl, 1);
plotPanel(ax1, ips, worstR2, models, markerNames, markerSizes, colors, ...
    'Inference speed (IPS)', 'Worst-state R^2', '(a) Accuracy vs speed', ...
    'xlog', true, 'ylog', false, 'clipYMin', 0.75, 'panel', 1);
xlim(ax1, paddedLogLimits(ips));
ylim(ax1, [0.75, 1.02]);

ax2 = nexttile(tl, 2);
plotPanel(ax2, params, worstR2, models, markerNames, markerSizes, colors, ...
    'Model complexity (parameters)', 'Worst-state R^2', '(b) Accuracy vs complexity', ...
    'xlog', true, 'ylog', false, 'clipYMin', 0.75, 'panel', 2);
xlim(ax2, paddedLogLimits(params));
ylim(ax2, [0.75, 1.02]);

ax3 = nexttile(tl, 3);
plotPanel(ax3, params, ips, models, markerNames, markerSizes, colors, ...
    'Model complexity (parameters)', 'Inference speed (IPS)', '(c) Speed vs complexity', ...
    'xlog', true, 'ylog', true, 'panel', 3);
xlim(ax3, paddedLogLimits(params));
ylim(ax3, paddedLogLimits(ips));

lgd = legend(ax1, models, 'Orientation', 'horizontal', 'Box', 'off', ...
    'FontName', paperFont, 'FontSize', legendFontSize, 'NumColumns', n);
lgd.Layout.Tile = 'north';

exportgraphics(fig, fullfile(outDir, 'fig7_benchmark_visualization_new.png'), 'Resolution', 600);
close(fig);

fprintf('Saved Fig. 7 benchmark visualization to:\n');
fprintf('  %s\n', fullfile(outDir, 'fig7_benchmark_visualization_new.png'));

function plotPanel(ax, x, y, labels, markerNames, markerSizes, colors, xLabel, yLabel, titleText, opts)
    arguments
        ax
        x
        y
        labels
        markerNames
        markerSizes
        colors
        xLabel
        yLabel
        titleText
        opts.xlog logical = false
        opts.ylog logical = false
        opts.clipYMin double = -Inf
        opts.panel double = 1
    end

    hold(ax, 'on');
    for ii = 1:numel(labels)
        yPlot = max(y(ii), opts.clipYMin);
        markerName = markerNames{ii};
        if y(ii) < opts.clipYMin
            markerName = 'v';
        end
        plot(ax, x(ii), yPlot, markerName, ...
            'MarkerSize', markerSizes(ii), ...
            'MarkerFaceColor', colors(ii, :), ...
            'MarkerEdgeColor', [0.10, 0.10, 0.10], ...
            'LineWidth', 0.65, ...
            'DisplayName', labels{ii});
    end

    if opts.xlog
        set(ax, 'XScale', 'log');
    end
    if opts.ylog
        set(ax, 'YScale', 'log');
    end

    xlabel(ax, xLabel);
    ylabel(ax, yLabel);
    title(ax, titleText, 'FontWeight', 'normal');
    styleAxes(ax);
end

function v = logInterp(lim, f)
    v = 10 ^ (log10(lim(1)) + f * (log10(lim(2)) - log10(lim(1))));
end

function lim = paddedLogLimits(values)
    values = values(isfinite(values) & values > 0);
    lo = min(values);
    hi = max(values);
    lim = [lo / 1.8, hi * 1.8];
end

function styleAxes(ax)
    set(ax, 'FontName', 'Arial', 'FontSize', 8, ...
        'LineWidth', 0.65, 'TickDir', 'out', 'Box', 'off', ...
        'Layer', 'top', 'XColor', [0, 0, 0], 'YColor', [0, 0, 0]);
    ax.TickLength = [0.018, 0.018];
    grid(ax, 'off');
end
