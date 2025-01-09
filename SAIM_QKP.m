clear all
close all
clc

%% This script solves Quadratic Knapsack Problems (QKP) using a Self-Adaptive Ising Machine based on probabilistic bits (p-bits)

% Author: Corentin Delacour, Electrical and Computer Engineering Department, University of California Santa Barbara
% delacour@ucsb.edu


%% Reading input instance 

problem=importdata("../QKP/jeu_300_25_1.txt")

density=0.25 % density of the W-matrix for quadratic interactions

opt=29140; % optimal value available at https://cedric.cnam.fr/~soutif/QKP/
f_ip=-opt % maximization problem converted into a minimization problem

% Number of Knapsack items (binary variables)
N=problem.data(1);

% Ising external field corresponding to the linear part of the cost function
for i=1:N
    hx(i)=-problem.data(1+i);% negative sign for minimization
end

% Ising weight matrix corresponding to the quadratic part of the cost function
k=N+1;
for i=1:N-1
    for j=i+1:N
        k=k+1;
        Wx(i,j)=-problem.data(k)/2; % factor 0.5 because there is a double contribution from Wij=Wji
        Wx(j,i)=Wx(i,j);
    end
end
k=k+2; % to skip 0 from input text file

% Inequality constraints Ax * X <= Bx
Bx=problem.data(k);

for i=1:N
    k=k+1;
    Ax(i)=problem.data(k); 
end

N_ineq=length(Bx);
N_constraints=N_ineq;

%% Transforming inequality constraints to equality using slack variables

s_max=zeros(N_ineq,1); % max slack value
N_slack_vector=zeros(N_ineq,1); % Number of binary slack variables needed per inequality

for i=1:N_ineq
    %checking max slack variable value: looking at negative coefficients
    %from A(i,:)
    for j=1:N
        if (Ax(i,j)<0)
            s_max(i)=s_max(i)-Ax(i,j);
        end
    end
    s_max(i)=Bx(i)+s_max(i);
    if s_max(i)~=0
        N_slack_vector(i)=floor(log2(abs(s_max(i)))+1);
    end

end

N_slack=sum(N_slack_vector); % total number of binary slack variables

% Building the binary decomposition of slack variables
slack_matrix=zeros(N_ineq,N_slack);
jump=0;

for i=1:N_ineq
    power=0;
    if N_slack_vector(i)~=0
        for j=1:N_slack_vector(i)
            slack_matrix(i,j+jump)=2^power;
            power=power+1;
        end
        jump=jump+N_slack_vector(i);
    end
end

% Total number of variables
Nt=N+N_slack;

% New matrices for equality constraints
Ax=[Ax slack_matrix];
hx=[hx zeros(1,N_slack)];
Wx=[Wx zeros(N,N_slack);
    zeros(N_slack,N+N_slack)];

max_Ax=max(abs(Ax),[],'all'); % Used for normalization later
max_Bx=max(abs(Bx),[],'all');
max_hx=max(abs(hx),[],'all');
max_Wx=max(abs(Wx),[],'all');

max_coeff_obj=max([max_hx max_Wx]);

%% Normalizing matrices

% Constraints
Ax=Ax/max(max_Ax,max_Bx);
Bx=Bx/max(max_Ax,max_Bx);

% Cost function
hx=hx/max_coeff_obj;
Wx=Wx/max_coeff_obj;


%% Conversion unipolar variables x={0;1} -> bipolar spins S={-1;1} for Ising machines

% Cost function
h=hx/2+0.5*(Wx*ones(Nt,1))';
W=Wx/4;

% Constraints
A=Ax;
B=2*Bx-Ax*ones(Nt,1);



%% Configuration of the self-adaptive Ising machine based on probablistic bits

% Penalty coefficient
P=2*Nt*density*ones(N_constraints,1);

% Building new W including penalty
W_new=W;
for m=1:N_constraints
    W_new=W_new+P(m)*A(m,:)'*A(m,:);
end

% Removing diagonal terms since S^2=1=constant
for i=1:Nt
    W_new(i,i)=0;
end

h_new=h';
for m=1:N_constraints
   h_new=h_new-(2*P(m)*B(m)*A(m,:)');
end

% Number of Monte Carlo sweeps
% (a MCS corresponds to the evaluation of all Nt variables)
kmax=1e3;

% Maximum inverse temperature for the p-bit system
beta_max=10

% Initial Lagrange multiplier value
lam=zeros(N_constraints,1);

% Lagrange multiplier learning rate
eta=20

% Number of Lagrange multiplier updates
max_step=2000;


%% Running the p-bit-based self adaptive Ising machine


% Storing cost values
cost_feas=NaN*ones(kmax*max_step,1);
cost_feas_step=NaN*ones(max_step,1);
sample_time=NaN*ones(max_step,1);
cost=zeros(kmax*max_step,1);

% Storing spin and Lagrange multiplier values
S=zeros(Nt,kmax);
I=zeros(Nt,1);
lag_store=zeros(N_constraints,kmax*max_step);

% Counting the number of MCS
t=0;

% A step is a Lagrange multiplier update
for step=1:max_step

    % Energy landscape update
    
    % Building h
    h_new=h';
    for m=1:N_constraints
       h_new=h_new+(-(2*P(m)*B(m)*A(m,:)')+ A(m,:)'*lam(m));
    end

    % random initial spin state
    S_temp=sign(randn(Nt,1));

    % Generating samples to find the minimum of the Lagrange function at given step
    for k=1:kmax
        t=t+1;

        % Running Simulated Annealing with linear beta schedule at each
        % step
        beta=beta_max*k/kmax;
    
        % p-bit Monte Carlo sweep
        for i=1:Nt
            I(i)=-2*W_new(i,:)*S_temp-h_new(i); % input current
            S_temp(i)=sign(tanh(beta*I(i))-2*rand(1,1)+1); % spin update
        end

        % Evaluation of the cost function
        cost(t)=(0.5*(S_temp'+1)*Wx*0.5*(S_temp+1)+hx*0.5*(S_temp+1))*max_coeff_obj;
    
        % Storing the Lagrange multiplier value
        lag_store(:,t)=lam;
        
        % Checking initial inequality constraint
        condition_feasibility=(Ax(:,1:N)*(S_temp(1:N)+1)*0.5<=Bx);

        if condition_feasibility
            cost_feas(t)=cost(t);
        end
    

    end

    % Recording feasible solutions at the end of each SA run
    if condition_feasibility
          if abs(cost_feas(t)-f_ip)<1
                fprintf("Optimal solution found: %f\n",cost_feas(t));
          end
          sample_time(step)=t;
          cost_feas_step(step)=cost_feas(t);
    end

    % Lagrange multiplier update
    lam=lam+eta*(A*S_temp-B);
    


end

% Compute feasibility over all MCS
feasibility_ratio=(t-sum(isnan(cost_feas(1:t))))/t

% Compute feasibility over measured samples ONLY (at the end of each SA
% run)
feasibility_ratio_samples=(step-sum(isnan(cost_feas_step(1:step))))/step

% s = struct("ip_time",ip_time,"cost_feas_step",cost_feas_step,"feas_ratio",feasibility_ratio_samples);
% save(sprintf("100_10_0%d.mat",instance),"-fromstruct",s);


figure
subplot(1,2,1)
plot(1:1:t,cost(1:t),'LineWidth',3) % Plot SAIM cost across all MCS
xlabel('MCS')
ylabel('cost')
hold on
plot(kmax:kmax:kmax*max_step,cost(kmax:kmax:kmax*max_step),'*','LineWidth',5,'Color','red') % Plot SAIM cost measured at the end of each SA step
plot(sample_time,cost_feas_step,'*','LineWidth',5,'Color','green') % Plot feasible measured cost
plot([0 t],f_ip*[1 1],'--','LineWidth',2,'Color','black') % Optimal cost value
legend('cost','Measured unfeasible','Measured feasible','OPT')
set(gca,'FontSize',20)

% Plot Lagrange multiplier dynamics
subplot(1,2,2)
plot(1:1:t,lag_store,'LineWidth',3)
ylabel('\lambda')
hold on
xlabel('MCS')
set(gca,'FontSize',20)
grid on

%% Post processing

% Measuring the feasibility and optimality ratios among MEASURED samples
cost_feas_avg=0;
k=0;
reach_opt=0;
for i=1:length(cost_feas_step)

    if (not(isnan(cost_feas_step(i)))) % if measured sample is feasible
        cost_feas_avg=cost_feas_avg+cost_feas_step(i);
        k=k+1;
    end

    if abs(cost_feas_step(i)-f_ip)<1
        reach_opt=reach_opt+1;
    end
end


cost_feas_avg=cost_feas_avg/k;

reach_optimum=100*reach_opt/k

accuracy=100*cost_feas_avg/f_ip

best_accuracy=100*min(cost_feas_step)/f_ip

MCS=max_step*kmax

