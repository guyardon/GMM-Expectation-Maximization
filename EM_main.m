%% Expectation Maximization Demo

clear all
close all
clc

%% Define ground truth GMM parameters for simulation 
% Choose # Random Variables (M)
% Choose # Gaussians (K)
% Draw random mu, sigma, phi
% Draw samples from the multivariate GMM distribution

M = 2;
K = 3;
limits = [-5,5];

[mu, sigma, phi] = draw_GMM_parameters(K,M,limits);
GMM              = gmdistribution(mu,sigma,phi);
data             = random(GMM,1000);


% plot data

figure; set(gcf,'WindowStyle','docked');
scatter(data(:,1), data(:,2),'filled');
grid on;


%% Run EM algorithm
% Define hyperparameters: K, tol, n_iter, 
% Initialize estimates for mu, sigma, phi
% Call EM function

K_est     = 3;
tol       = 1e-6;
n_iter    = 50;
plot_flag = true;

[mu0, sigma0, phi0] = draw_GMM_parameters(K,M,limits);


[model_params] = ...
    ...
    EM_algorithm(data, mu0, sigma0, phi0, K_est, tol, n_iter, plot_flag);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [model_params] = ...
    ...
    EM_algorithm(data, mu0, sigma0, phi0, K_est, tol, n_iter, plot_flag)

model_params.mu    = mu0;
model_params.sigma = sigma0;
model_params.phi   = phi0;

hyper_params.K     = K_est;
N = size(data,1);

likelihood = zeros(1,n_iter);

if plot_flag
    figure; set(gcf,'WindowStyle','docked');
    ax1 = subplot(1,2,1);
    ax2 = subplot(1,2,2);
    
    grid(ax1,'on');
    xlabel(ax1,'X_{1}');
    ylabel(ax1,'X_{2}');
    title(ax1,'Expectation-Maximization of the GMM');
    
    grid(ax2,'on');
    xlabel(ax2,'Iteration #');
    ylabel(ax2,'Log-Likelihood');
    title(ax2,'Log-Likelihood');
    
    x = linspace(-20,20,1000); %// x axis
    y = linspace(-20,20,1000); %// y axis
    [X ,Y] = meshgrid(x,y); %// all combinations of x, y
end

for iter = 1 : n_iter
    
   % Expectation step - per sample
   hidden_params = expectation(data, model_params, hyper_params);
   
   % Maximization step - per gaussian
   model_params = maximization(data, model_params, ...
       ...
       hidden_params,hyper_params);
     
   %% Calculate Likelihood
   
   mu_est    = model_params.mu;
   sigma_est = model_params.sigma;
   phi_est   = model_params.phi;
   
   likelihood(iter) = 0;
   for n = 1 : N
       for k = 1 : K_est
           likelihood(iter) = likelihood(iter) + ...
               (phi_est(k) * mvnpdf(data(n,:),mu_est(k,:),sigma_est(:,:,k)));
       end
   end
   
   if iter > 2
       converged = norm(log(likelihood(iter)) - log(likelihood(iter-1))) < tol;
       if converged, return, end
   end
   
   %% Plot Current Iteration
   
   if plot_flag
       cla(ax1); hold(ax1,'all');
       
       scatter(ax1,data(:,1), data(:,2),[],'k.');
       
       for k = 1 : K_est
           mu = mu_est(k,:);
           sigma = sigma_est(:,:,k); %// data
           Z = mvnpdf([X(:), Y(:)],mu,sigma); %// compute Gaussian pdf
           Z = reshape(Z,size(X)); %// put into same size as X, Y
           contour(ax1,X,Y,Z,'LineWidth',1);
           colorbar;
       end
       
       cla(ax2); hold(ax2,'all');
       plot(ax2,1:iter,log(likelihood(1:iter)),'-o','LineWidth',2);
       drawnow;
       pause(0.2);
   end
   
end

figure; set(gcf,'WindowStyle','docked');
plot(1:n_iter,log(likelihood),'-o','LineWidth',2);
grid on;
xlabel('Iteration #');
ylabel('Log-Likelihood');
title('Log-Likelihood');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function hidden_params = ...
    expectation(data, model_params, hyper_params)

    % get sizes
    K = hyper_params.K;
    N = size(data,1);
    M = size(data,2);


    % initialize hidden variables in Q
    Q = zeros(K,N);

    for k = 1 : K

        % get model parameters for current gaussian
        mu    = model_params.mu;
        sigma = model_params.sigma;
        phi   = model_params.phi;

        for n = 1 : N

            inv_norm_factor = 0;

            for kk = 1 : K
                inv_norm_factor = inv_norm_factor + phi(kk) * ...
                    ...
                    mvnpdf(data(n,:),mu(kk,:),sigma(:,:,kk));
            end

            Q(k,n) = phi(k) * ...
                    ...
                    mvnpdf(data(n,:),mu(k,:),sigma(:,:,k)) ./ ...
                    inv_norm_factor;

        end
    end

    hidden_params.Q = Q;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function model_params = maximization(data, model_params, ...
       ...
       hidden_params,hyper_params)
   
   % get sizes
   K = hyper_params.K;
   N = size(data,1);
   M = size(data,2);
   
   % get latent variables in Q
   Q = hidden_params.Q;
   
   % initialize outputs
   mu_est    = zeros(K,M);
   sigma_est = zeros(M,M,K);
   phi_est   = zeros(K,1);
   
   for k = 1 : K
       phi_est(k)       = 1/N * sum( Q(k,:) );
       mu_est(k,:)      = Q(k,:)* data  ...
                            ./ sum( Q(k,:) );
       
       sigma_k = zeros(M,M);                    
       for n = 1 : N
           X_n = data(n,:) - mu_est(k,:); % 1 x m
           sigma_k = sigma_k + Q(k,n) * (X_n'*X_n) ./ sum( Q(k,:) );
       end
       
       sigma_est(:,:,k) = sigma_k;
       
   end
   
   % save updated model parameters
   model_params.mu    = mu_est;
   model_params.sigma = sigma_est;
   model_params.phi   = phi_est;
   
   
   

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
function [mu, sigma, phi] = draw_GMM_parameters(K,M,limits)

    lower_lim = limits(1);
    upper_lim = limits(2);

    mu               = lower_lim + (upper_lim - lower_lim)*rand(K,M);
    sigma            = lower_lim + (upper_lim - lower_lim)*rand(M,M,K);
    
    mu(1,:) = mu(1,:);
    mu(2,:) = mu(2,:);
    mu(3,:) = mu(3,:);
    
    for k = 1 : K
        sigma(2,1,k) = sigma(1,2,k);
        sigma(:,:,k) = sigma(:,:,k)'*sigma(:,:,k);
    end

    random_weights   = rand(K,1);
    phi              = random_weights ./ sum(random_weights);

end




