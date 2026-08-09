%% BART on the Friedman test function
% Demonstrates fitting, prediction, uncertainty quantification and a simple
% variable importance summary.

clear; close all;
rng(1, 'twister');

%% Simulate data
n = 750;
p = 10;                      % only the first five features matter
X = rand(n, p);
f = @(X) 10*sin(pi*X(:,1).*X(:,2)) + 20*(X(:,3)-0.5).^2 + 10*X(:,4) + 5*X(:,5);
sigma = 1.0;
y = f(X) + sigma*randn(n, 1);

% Hold out a test set
isTest = false(n, 1);
isTest(randperm(n, 250)) = true;
XTrain = X(~isTest, :);  yTrain = y(~isTest);
XTest  = X(isTest, :);   yTest  = y(isTest);

%% Fit
tic;
model = stochtree.bart(XTrain, yTrain, ...
    'XTest', XTest, ...
    'NumGFR', 10, ...        % XBART warm start, shared by all chains
    'NumBurnin', 200, ...    % per chain; matters when running several
    'NumMCMC', 250, ...      % retained draws per chain
    'NumChains', 4, ...      % 1000 draws in total
    'NumTrees', 100, ...
    'RandomSeed', 42);
fprintf('Fit in %.1f s\n', toc);
model.summary();

%% Accuracy
yhatTest = mean(model.YHatTest, 2);
rmse = sqrt(mean((yTest - yhatTest).^2));
r2 = 1 - sum((yTest - yhatTest).^2) / sum((yTest - mean(yTest)).^2);
fprintf('Test RMSE %.3f, R^2 %.3f (true noise sd %.2f)\n', rmse, r2, sigma);

%% Coverage of the 95% posterior interval for the conditional mean
ci = quantile(model.YHatTest, [0.025 0.975], 2);
fTest = f(XTest);
coverage = mean(fTest >= ci(:,1) & fTest <= ci(:,2));
fprintf('95%% interval coverage of the true mean function: %.1f%%\n', 100*coverage);

%% Convergence across chains
model.convergenceDiagnostics()

%% Plots
figure('Position', [100 100 1100 350]);

subplot(1,3,1);
errorbar(fTest, yhatTest, yhatTest - ci(:,1), ci(:,2) - yhatTest, ...
    'o', 'MarkerSize', 3, 'CapSize', 0, 'Color', [0.6 0.6 0.85]);
hold on; plot(xlim, xlim, 'k--', 'LineWidth', 1);
xlabel('True f(x)'); ylabel('Posterior mean'); title('Test set fit');
axis square; grid on;

subplot(1,3,2);
plot(model.chainMatrix(model.Sigma2Samples), 'LineWidth', 0.8);
yline(sigma^2, 'r--', 'LineWidth', 1.5);
xlabel('Draw within chain'); ylabel('\sigma^2');
title(sprintf('\\sigma^2 by chain (R-hat %.3f)', ...
    stochtree.rhat(model.Sigma2Samples, model.ChainIndex)));
grid on;

subplot(1,3,3);
counts = model.variableSplitCounts();
bar(counts);
xlabel('Covariate'); ylabel('Split count');
title('Variable importance (first 5 are real)'); grid on;

sgtitle('stochtree.bart on the Friedman function');
