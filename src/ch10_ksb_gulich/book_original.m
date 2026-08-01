% VERBATIM transcription — DO NOT EDIT.
%
% Source: "Parameter Identification of Empirical Models for Head Estimation of an
%          Electrical Submersible Pump Handling Viscous Fluids"
%          H. Hadabi et al.
%          in: Smart Diagnostics and Predictive Maintenance, ed. Aydin Azizi, Springer.
%          Printed pages 277-280  (PDF pages 284-287)
%
% Appendix 1: MATLAB Code Scripts.
% Requires MATLAB + Optimization Toolbox, and three data files that the book does not
% ship. A validated Python port of the model core is in ksb_gulich_python.py.

%% ===============================================================================
%  1. Nelder-Mead Optimization (fminsearch)            (printed p.277 / PDF p.284)
%  ===============================================================================

%% --- PARAMETER OPTIMIZATION USING NELDER-MEAD (fminsearch) ---
% Optimizes 9 empirical parameters (KSB + Gulich) to minimize prediction error

% Initial guess: [KSB params (6) + Gulich params (3)]
theta0 = [10.41, 0.039, 0.113, 5.323, 0.390, 0.643, 0.507, 12.09, 0.709];

% Optimization options
options = optimset('Display','iter','MaxIter',2000,'MaxFunEvals',5000);

% Load required data
T = readtable('TogetCorrectionFactors.xlsx');
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);
expTrain = readtable('TrainSet.xlsx');
waterData = readtable('Water_Combined_Head_vs_Flow_Data__All_RPMs_.csv');

% Run optimization
[theta_opt, ~] = fminsearch(@(theta) modelError(theta, T, expTrain, waterData, 'Combined'), theta0, options);

% Display results
disp('Optimized Parameters (Nelder-Mead):');
disp(theta_opt);

%% --- MODEL ERROR FUNCTION ---
function [rmse] = modelError(theta, T, expData, waterData, modelType)
    % Extract parameters
    a = theta(1); b = theta(2); c = theta(3);
    d = theta(4); e = theta(5); f = theta(6);
    a_g = theta(7); b_g = theta(8); c_g = theta(9);

    % Constants
    r = 0.056; g = 9.81; bpd2m3s = 0.00000184013;

    all_true = []; all_pred = [];

    for i = 1:height(T)
        rpm = T.RPM(i); visc = T.Viscosity(i); omega_s = T.omega_s(i);
        Qw_BEP = T.Q_BEP_BPD(i) * bpd2m3s;
        Hw_BEP = T.H_BEP_ft(i) * 0.3048;
        Qw = T.Q_BPD(i) * bpd2m3s;
        nu = (visc / 0.898) * 1e-6;
        omega = 2 * pi * rpm / 60;

        % KSB Model
        B_HI = (480 * sqrt(nu)) / (Qw_BEP^0.25 * (g * Hw_BEP)^0.125);
        B = exp(log(B_HI) + 0.5 * (log(a) - log(52.933) - log(omega_s)));
        CQ_KSB = exp(b * B * log(a / (52.933 * omega_s)) - c * (log10(B))^d);
        xi = 1 - 0.014 * (B_HI - 1) * ((Qw / Qw_BEP) - 1);
        CH_KSB = (e + f * CQ_KSB) * xi;

        % Gulich Model
        Re_omega = (omega * r^2) / nu;
        Re_G = Re_omega * omega_s^a_g;
        x = -exp(log(b_g) - c_g * (log(Re_omega) + a_g * log(omega_s)));
        CH_BEP = Re_G^x;
        CQ_G = CH_BEP;
        CH_G = 1 - (1 - CH_BEP) * (Qw / Qw_BEP)^0.75;

        % Get data subsets
        W = waterData(waterData.RPM == rpm, :);
        E = expData(expData.RPM == rpm & expData.Viscosity == visc, :);
        if isempty(W) || isempty(E), continue; end

        % Predictions
        if strcmp(modelType, 'KSB') || strcmp(modelType, 'Combined')
            Q_pred = W.Flow_BPD * CQ_KSB;
            H_pred = W.Head_ft * CH_KSB;
            H_interp = interp1(Q_pred, H_pred, E.Flow_BPD, 'linear', 'extrap');
            all_pred = [all_pred; H_interp];
            all_true = [all_true; E.Head_ft];
        end
        if strcmp(modelType, 'Gulich') || strcmp(modelType, 'Combined')
            Q_pred = W.Flow_BPD * CQ_G;
            H_pred = W.Head_ft * CH_G;
            H_interp = interp1(Q_pred, H_pred, E.Flow_BPD, 'linear', 'extrap');
            all_pred = [all_pred; H_interp];
            all_true = [all_true; E.Head_ft];
        end
    end

    % Compute RMSE
    err = all_true - all_pred;
    rmse = sqrt(mean(err.^2));
end


%% ===============================================================================
%  2. Levenberg-Marquardt Optimization (lsqnonlin)     (printed p.279 / PDF p.286)
%  ===============================================================================

%% --- PARAMETER OPTIMIZATION USING LEVENBERG-MARQUARDT (lsqnonlin) ---
% Optimizes 9 empirical parameters using nonlinear least squares

% Load datasets
T = readtable('TogetCorrectionFactors.xlsx', 'VariableNamingRule', 'modify');
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);
expTrain = readtable('TrainSet.xlsx');
waterData = readtable('Water_Combined_Head_vs_Flow_Data__All_RPMs_.csv');

% Standardize variable names
expTrain.Properties.VariableNames{'Speed'} = 'RPM';
expTrain.Properties.VariableNames{'Flowrate'} = 'Flow_BPD';
expTrain.Properties.VariableNames{'Head'} = 'Head_ft';

% Initial parameter guesses
ksb_init    = [10.41, 0.039, 0.113, 5.323, 0.390, 0.643];
gulich_init = [0.507, 12.09, 0.709];
theta0 = [ksb_init, gulich_init];

% Bounds
lb = [1, 0, 0, 0, 0, 0, 0, 1, 0];
ub = [50, 1, 5, 10, 5, 5, 2, 20, 2];

% Options
opts = optimoptions('lsqnonlin', 'Display', 'iter', 'MaxIterations', 1000);

% Residual function
residualFun = @(theta) modelResiduals(theta, T, expTrain, waterData);

% Run optimization
[theta_opt, resnorm] = lsqnonlin(residualFun, theta0, lb, ub, opts);

% Display results
disp('Optimized Parameters (Levenberg-Marquardt):');
disp(theta_opt);

%% --- MODEL RESIDUAL FUNCTION ---
function residuals = modelResiduals(theta, T, expData, waterData)
    a = theta(1); b = theta(2); c = theta(3);
    d = theta(4); e = theta(5); f = theta(6);
    a_g = theta(7); b_g = theta(8); c_g = theta(9);

    r = 0.056; g = 9.81; bpd2m3s = 0.00000184013;
    residuals = [];

    for i = 1:height(T)
        rpm = T.RPM(i); visc = T.Viscosity(i); omega_s = T.omega_s(i);
        Qw_BEP = T.Q_BEP_BPD(i) * bpd2m3s;
        Hw_BEP = T.H_BEP_ft(i) * 0.3048;
        Qw = T.Q_BPD(i) * bpd2m3s;
        nu = (visc / 0.898) * 1e-6;
        omega = 2 * pi * rpm / 60;

        % KSB Correction
        B_HI = (480 * sqrt(nu)) / (Qw_BEP^0.25 * (g * Hw_BEP)^0.125);
        B = exp(log(B_HI) + 0.5 * (log(a) - log(52.933) - log(omega_s)));
        CQ_KSB = exp(b * B * log(a / (52.933 * omega_s)) - c * (log10(B))^d);
        xi = 1 - 0.014 * (B_HI - 1) * ((Qw / Qw_BEP) - 1);
        CH_KSB = (e + f * CQ_KSB) * xi;

        % Gulich Correction
        Re_omega = (omega * r^2) / nu;
        Re_G = Re_omega * omega_s^a_g;
        x = -exp(log(b_g) - c_g * (log(Re_omega) + a_g * log(omega_s)));
        CH_BEP = Re_G^x;
        CQ_G = CH_BEP;
        CH_G = 1 - (1 - CH_BEP) * (Qw / Qw_BEP)^0.75;

        W = waterData(waterData.RPM == rpm, :);
        E = expData(expData.RPM == rpm & expData.Viscosity == visc, :);
        if isempty(W) || isempty(E), continue; end

        Q_KSB = W.Flow_BPD * CQ_KSB;
        H_KSB = W.Head_ft * CH_KSB;
        Q_G = W.Flow_BPD * CQ_G;
        H_G = W.Head_ft * CH_G;

        Hp_K = interp1(Q_KSB, H_KSB, E.Flow_BPD, 'linear', 'extrap');
        Hp_G = interp1(Q_G, H_G, E.Flow_BPD, 'linear', 'extrap');

        % Residuals: average of both models
        res = ((Hp_K + Hp_G) / 2) - E.Head_ft;
        residuals = [residuals; res];
    end
end
