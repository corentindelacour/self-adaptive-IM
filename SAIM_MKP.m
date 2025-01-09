clear all
close all
clc

%% This script solves Multidimensional Knapsack Problems (MKP) using a Self-Adaptive Ising Machine based on probabilistic bits (p-bits)

% Author: Corentin Delacour, Electrical and Computer Engineering Department, University of California Santa Barbara
% delacour@ucsb.edu


%% Reading MKP input instance in .mps format
% Variables are initially binary x={0;1}

instance=2
problem=mpsread('MKP/class1/100-5-0'+string(instance)+'.mps');

% Knapsack cost function:
Cx=-full(problem.f)';% (-) sign converts maximization into minimization problem
max_Cx=max(abs(Cx),[],'all'); % used later for normalization


% Number of Knapsack items (variables):
N=length(problem.lb); 
integer_variable_list=problem.intcon;


% Constraint inequalities Ax * X <= Bx
Ax=full(problem.Aineq);
Bx=full(problem.bineq);

N_ineq=length(Bx);
N_constraints=N_ineq;


%% Introducing binary slack variables

s_max=zeros(N_ineq,1); % Measures the max slack value for each inequality
N_slack_vector=zeros(N_ineq,1); % Stores the number of binary slack variables for each inequality
for i=1:N_ineq
    % checking max slack variable value: looking at negative coefficients from A(i,:)
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

N_slack=sum(N_slack_vector);

% Constructing the binary decomposition of slack variables
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

% Total number of variables, including binary slack variables
Nt=N+N_slack;

% Slack variables transform inequality constraints into equality
% constraints as Ax * X = Bx with the new matrices defined as follows:
Ax=[Ax slack_matrix];
Cx=[Cx zeros(1,N_slack)];

max_Ax=max(abs(Ax),[],'all'); % used later for normalization
max_Bx=max(abs(Bx),[],'all');



%% Normalizing matrices

% Normalization of constraint equation
Ax=Ax/max(max_Ax,max_Bx);
Bx=Bx/max(max_Ax,max_Bx);

% Normalization of cost function
Cx=Cx/max_Cx;

%% Conversion of unipolar variables x={0;1} -> bipolar spins S={-1;1} for Ising machines

C=Cx/2;
A=Ax;
B=2*Bx-Ax*ones(Nt,1);


%% Find MKP optimal value using Branch & Bound algorithm (B&B): Ground truth
options = optimoptions('intlinprog','MaxTime',1200)

tic
[ x_ip , f_ip ] = intlinprog ( Cx, [1:Nt],[],[] ,Ax, Bx,zeros(Nt,1),ones(Nt,1)); % MKP is expressed as an Integer Linear Program
f_ip=f_ip*max_Cx; % optimal value expressed in its original form (unormalized)

% B&B runtime
ip_time=toc


%% Configuring p-bit-based Self-Adaptive Ising machine

% Penalty coefficient (kept fixed during the dynamics)
% Here, each constraint has the same penalty coefficient
P=(5*Nt*2/(Nt+1))*ones(N_constraints,1);

% Building W matrix: pairwise spin coupling
W_new=zeros(Nt,Nt);
for m=1:N_constraints
    W_new=W_new+P(m)*A(m,:)'*A(m,:);
end

% removing diagonal terms that are useless for the optimization process
% (since S^2=1=constant)
for i=1:Nt
    W_new(i,i)=0;
end

% Number of Monte Carlo sweeps (1 MCS= update of all Nt variables)
kmax=1000;

% Lagrange multiplier learning rate
eta=0.05

% Number of Lagrange multiplier updates
max_step=5000;


%% Running the p-bit-based Self-Adaptive Ising machine

% Storing S dynamics
S=zeros(Nt,kmax);
I=zeros(Nt,1);

% Initial Lambda values
lam=zeros(N_constraints,1);

% Storing cost and Lagrange multiplier values
cost_feas=NaN*ones(kmax*max_step,1);
cost_feas_step=NaN*ones(max_step,1);
sample_time=NaN*ones(max_step,1);
cost=zeros(kmax*max_step,1);
lag_store=zeros(N_ineq,kmax*max_step);

t=0; % Counts each MCS

% a step is a Lagrange multiplier update
for step=1:max_step

    % Energy landscape update
    % Building external field h
    h_new=C';
    for m=1:N_constraints
       h_new=h_new+(-(2*P(m)*B(m)*A(m,:)')+ A(m,:)'*lam(m)); % External fields for the Ising spins
    end

    % Random Initial spin state before running the Ising machine
    S_temp=sign(randn(Nt,1));

    % Generating samples to find minimum of Lagrange function at given step
    for k=1:kmax
        t=t+1;

        %running SA with linear beta increase for each step
        beta=50*k/kmax;
    
        % p-bit Monte Carlo sweep
        for i=1:Nt
            I(i)=-2*W_new(i,:)*S_temp-h_new(i); % input current
            S_temp(i)=sign(tanh(beta*I(i))-2*rand(1,1)+1); % spin update
        end
    
        % Monitoring cost function
        cost(t)=Cx*0.5*(S_temp+1)*max_Cx;
    
        % Storing Lagrange multipliers
        lag_store(:,t)=lam;
        
        % Checking initial inequality constraints
        condition_feasibility=(Ax(:,1:N)*(S_temp(1:N)+1)*0.5<=Bx);%

        if condition_feasibility
            cost_feas(t)=cost(t);
        end
    
    end

    % Recording feasible solutions at the end of each SA run (step)
    if condition_feasibility
        if abs(cost_feas(t)-f_ip)<1
            fprintf("Optimal solution found:%f\n",cost_feas(t));
        end
        sample_time(step)=t;
        cost_feas_step(step)=cost_feas(t);
    end

    % Update of Lagrange multipliers
    lam=lam+eta*(A*S_temp-B);


end

% Measuring feasibility ratio across ALL samples
feasibility_ratio=(t-sum(isnan(cost_feas(1:t))))/t

% Measuring feasibility ratio across MEASURED samples (at the end of SA)
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

