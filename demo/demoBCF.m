%% Bayesian Causal Forest with a heterogeneous treatment effect
% Simulates confounded observational data where the treatment effect varies
% with a covariate, then recovers the CATE surface and the ATE.

clear; close all;
rng(2);

%% Simulate confounded data
n = 2000;
p = 5;
X = rand(n, p);

% Propensity depends on X1, so treatment assignment is confounded
propensity = 0.15 + 0.7 * X(:,1);
Z = double(rand(n, 1) < propensity);

muX  = 3*X(:,1) + 2*X(:,2) - X(:,3);       % prognostic function
tauX = 1 + 2*X(:,2);                        % heterogeneous treatment effect
sigma = 0.5;
y = muX + tauX .* Z + sigma * randn(n, 1);

fprintf('True ATE: %.4f\n', mean(tauX));

% A naive difference in means is badly confounded
naive = mean(y(Z==1)) - mean(y(Z==0));
fprintf('Naive difference in means: %.4f (biased by confounding)\n', naive);

%% Fit BCF
% The true propensity is supplied here. Omit 'PropensityTrain' to have it
% estimated internally with a small BART model.
tic;
model = stochtree.bcf(X, Z, y, ...
    'PropensityTrain', propensity, ...
    'NumGFR', 10, ...
    'NumBurnin', 100, ...
    'NumMCMC', 500, ...
    'RandomSeed', 42);
fprintf('Fit in %.1f s\n', toc);

%% ATE
ate = model.ateSummary(0.95);
fprintf('BCF ATE posterior mean %.4f, 95%% CI [%.4f, %.4f]\n', ...
    ate.mean, ate.interval(1), ate.interval(2));
fprintf('Covers the truth: %d\n', ...
    mean(tauX) >= ate.interval(1) && mean(tauX) <= ate.interval(2));

%% CATE recovery
tauHat = mean(model.TauHatTrain, 2);
tauCI = quantile(model.TauHatTrain, [0.025 0.975], 2);
fprintf('CATE RMSE %.4f, correlation %.4f\n', ...
    sqrt(mean((tauHat - tauX).^2)), corr(tauHat, tauX));

%% Plots
figure('Position', [100 100 1100 350]);

subplot(1,3,1);
scatter(tauX, tauHat, 8, 'filled', 'MarkerFaceAlpha', 0.25);
hold on; plot(xlim, xlim, 'k--', 'LineWidth', 1);
xlabel('True \tau(x)'); ylabel('Estimated \tau(x)');
title('CATE recovery'); axis square; grid on;

subplot(1,3,2);
[~, order] = sort(X(:,2));
fill([X(order,2); flipud(X(order,2))], [tauCI(order,1); flipud(tauCI(order,2))], ...
    [0.8 0.85 0.95], 'EdgeColor', 'none'); hold on;
plot(X(order,2), tauHat(order), 'b-', 'LineWidth', 1.2);
plot(X(order,2), tauX(order), 'r--', 'LineWidth', 1.2);
xlabel('X_2'); ylabel('\tau(x)');
title('Effect heterogeneity in X_2');
legend('95% CI', 'posterior mean', 'truth', 'Location', 'best'); grid on;

subplot(1,3,3);
histogram(model.ateSamples(), 30, 'FaceColor', [0.4 0.5 0.8]);
xline(mean(tauX), 'r--', 'LineWidth', 1.5);
xline(naive, 'k:', 'LineWidth', 1.5);
xlabel('ATE'); ylabel('Posterior draws');
title('ATE posterior'); legend('draws', 'truth', 'naive', 'Location', 'best');

sgtitle('stochtree.bcf on confounded observational data');
