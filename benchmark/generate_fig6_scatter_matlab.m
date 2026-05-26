% Generate paper-ready Fig. 6: benchmark prediction scatter plots.
% Style target: clean Nature/Science-like double-column figure, Arial 8 pt.

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

paperFont = 'Arial';
fontSize = 8;
smallFontSize = 6.4;

models = {'PhysRes', 'Linear_AR', 'MLP', 'ESN', 'LSTM'};
modelTitles = {'PhysRes', 'Linear AR', 'MLP', 'ESN', 'LSTM'};
modelColors = [
    0.80, 0.10, 0.10;  % PhysRes
    0.45, 0.45, 0.45;  % Linear AR
    0.00, 0.45, 0.70;  % MLP
    0.49, 0.18, 0.56;  % ESN
    0.00, 0.60, 0.50   % LSTM
];

vars = struct( ...
    'name', {'P', 'Tp', 'Tl'}, ...
    'actual', {'P_actual_kPa', 'Tp_actual_C', 'Tl_actual_C'}, ...
    'predPrefix', {'P', 'Tp', 'Tl'}, ...
    'rowLabel', {'Pressure (kPa)', 'Peltier temp. (^{\circ}C)', 'Liquid temp. (^{\circ}C)'}, ...
    'rmseUnit', {'kPa', '^{\circ}C', '^{\circ}C'} ...
);

% Double-column RA-L width. The 3 x 5 grid needs a low-height, compact layout.
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2, 2, 17.8, 10.8]);

nRows = numel(vars);
nCols = numel(models);
left = 0.070;
right = 0.018;
bottom = 0.090;
top = 0.085;
hGap = 0.020;
vGap = 0.065;
panelW = (1 - left - right - (nCols - 1) * hGap) / nCols;
panelH = (1 - bottom - top - (nRows - 1) * vGap) / nRows;

for r = 1:nRows
    xAll = T.(vars(r).actual)(idx);
    axisLim = physicalLimits(xAll);

    for c = 1:nCols
        axLeft = left + (c - 1) * (panelW + hGap);
        axBottom = 1 - top - r * panelH - (r - 1) * vGap;
        ax = axes(fig, 'Position', [axLeft, axBottom, panelW, panelH]);
        hold(ax, 'on');

        modelName = models{c};
        predCol = sprintf('%s_%s_%s', vars(r).predPrefix, modelName, unitSuffix(vars(r).name));
        yAll = T.(predCol)(idx);

        [rmse, r2] = metrics(xAll, yAll);
        inMask = isfinite(yAll) & yAll >= axisLim(1) & yAll <= axisLim(2);
        highMask = isfinite(yAll) & yAll > axisLim(2);
        lowMask = isfinite(yAll) & yAll < axisLim(1);
        outPct = 100 * (sum(highMask) + sum(lowMask)) / numel(yAll);

        plot(ax, axisLim, axisLim, '--', 'Color', [0.18, 0.18, 0.18], ...
            'LineWidth', 0.65);

        color = modelColors(c, :);
        pointColor = 0.65 * color + 0.35 * [1, 1, 1];
        if any(inMask)
            s = scatter(ax, xAll(inMask), yAll(inMask), 3.2, pointColor, 'filled');
            s.MarkerEdgeColor = 'none';
        end
        if any(highMask)
            scatter(ax, xAll(highMask), axisLim(2) * ones(sum(highMask), 1), ...
                5.5, color, '^', 'filled', 'MarkerEdgeColor', 'none');
        end
        if any(lowMask)
            scatter(ax, xAll(lowMask), axisLim(1) * ones(sum(lowMask), 1), ...
                5.5, color, 'v', 'filled', 'MarkerEdgeColor', 'none');
        end

        xlim(ax, axisLim);
        ylim(ax, axisLim);
        axis(ax, 'square');
        styleAxes(ax, paperFont, fontSize);

        if r == 1
            title(ax, modelTitles{c}, 'FontName', paperFont, 'FontSize', fontSize, ...
                'FontWeight', 'normal');
        end
        if c == 1
            ylabel(ax, sprintf('Predicted\n%s', vars(r).rowLabel), ...
                'FontName', paperFont, 'FontSize', fontSize);
        else
            ax.YTickLabel = [];
        end
        metricText = metricLabel(rmse, r2, outPct, vars(r).rmseUnit);
        text(ax, 0.05, 0.07, metricText, 'Units', 'normalized', ...
            'FontName', paperFont, 'FontSize', smallFontSize, ...
            'VerticalAlignment', 'bottom', 'Color', [0.08, 0.08, 0.08], ...
            'BackgroundColor', 'w', 'Margin', 0.8);
    end
end

exportgraphics(fig, fullfile(outDir, 'fig6_scatter_benchmark_new.png'), 'Resolution', 600);
exportgraphics(fig, fullfile(outDir, 'fig6_scatter_benchmark_new.pdf'), ...
    'ContentType', 'image', 'Resolution', 600);
close(fig);

fprintf('Saved Fig. 6 scatter plots to:\n');
fprintf('  %s\n', fullfile(outDir, 'fig6_scatter_benchmark_new.png'));
fprintf('  %s\n', fullfile(outDir, 'fig6_scatter_benchmark_new.pdf'));

function suffix = unitSuffix(varName)
    switch varName
        case 'P'
            suffix = 'kPa';
        otherwise
            suffix = 'C';
    end
end

function lim = physicalLimits(values)
    values = values(isfinite(values));
    lo = min(values);
    hi = max(values);
    pad = 0.075 * (hi - lo);
    if pad <= 0
        pad = 1;
    end
    lim = [lo - pad, hi + pad];
end

function [rmse, r2] = metrics(yTrue, yPred)
    mask = isfinite(yTrue) & isfinite(yPred);
    yTrue = yTrue(mask);
    yPred = yPred(mask);
    rmse = sqrt(mean((yPred - yTrue).^2));
    ssRes = sum((yTrue - yPred).^2);
    ssTot = sum((yTrue - mean(yTrue)).^2);
    r2 = 1 - ssRes / ssTot;
end

function textOut = metricLabel(rmse, r2, outPct, unitText)
    if abs(rmse) >= 1e4
        rmseText = sprintf('RMSE=%.1e %s', rmse, unitText);
    elseif abs(rmse) >= 100
        rmseText = sprintf('RMSE=%.0f %s', rmse, unitText);
    elseif abs(rmse) >= 10
        rmseText = sprintf('RMSE=%.1f %s', rmse, unitText);
    else
        rmseText = sprintf('RMSE=%.3g %s', rmse, unitText);
    end

    if r2 < -99
        r2Text = 'R^2<-99';
    elseif r2 > 0.995
        r2Text = sprintf('R^2=%.4f', r2);
    else
        r2Text = sprintf('R^2=%.3f', r2);
    end

    if outPct > 0
        textOut = sprintf('%s\n%s, out=%.0f%%', rmseText, r2Text, outPct);
    else
        textOut = sprintf('%s\n%s', rmseText, r2Text);
    end
end

function styleAxes(ax, fontName, fontSize)
    set(ax, 'FontName', fontName, 'FontSize', fontSize, ...
        'LineWidth', 0.65, 'TickDir', 'out', 'Box', 'off', ...
        'Layer', 'top', 'XColor', [0, 0, 0], 'YColor', [0, 0, 0]);
    ax.TickLength = [0.018, 0.018];
    grid(ax, 'off');
end
