function swarm_engagement_function_v2(L,N_att,N_def,range_def,rate_of_fire_def,attacker_prob_survival,vflag,min_dist,plotit,seed)

% models optimal swarm on swarm scenario
% Isaac Kaminer 06/202/2023
% clc, clear all;
% close all;
% set(groot,'defaulttextInterpreter','latex');
% set(groot,'defaultAxesTickLabelInterpreter','latex');
% set(groot,'defaultLegendInterpreter','latex');

% L = 1;
% N_att = number of attackers (1 - 100)
% N_def = number of defenders (1 - 100)
% range_def = range of defender weapons (1 - 20)
% rate_of_fire_def - fire rate of defender weapons (0.1 - 2)
% attacker_prob_survival - upper bound on the probability of survival of
% any attacker
% vflag = 0;
% min_dist - any number
% plotit - 0 no plots, 1 - plots defender/attacker trajectories
% seed  - any number


file_name = ['swarm_opt_results_v2/swarm_scaling_v2_L_' num2str(L) '_Na_'  num2str(N_att) '_Nd_' num2str(N_def) ...
    '_Rd_' num2str(range_def) '_ROFd_' num2str(rate_of_fire_def) ...
    '_PSa_' num2str(attacker_prob_survival) '_Vflag_' num2str(vflag) ...
    '_min_dist_' num2str(min_dist) '_seed_' num2str(seed) '.mat'];

if exist(file_name) && ~plotit
    return
end

rng(seed);

if exist('IKsim.txt', 'file')==2
    delete('IKsim.txt');
end

rng(seed)
N_attacker=N_att;
N_defender=N_def;
Defender_v_max=1;
do_plot=1;
final_fraction=0;
%seed=2
accel=1;
kill_range=1;

t=0;
%set acceleration ramp up
steps_to_accel=accel;

ramp_time=1/steps_to_accel;
Def_a=Defender_v_max*ramp_time;

Att_v_min=.05;
Att_v_max=0.4;


vm = Att_v_max-Att_v_min;

v = Att_v_min+vm.*rand(N_attacker,1);
theta = pi/2.*rand(N_attacker,1);


Att_vel(:,1)=v.*cos(theta);    %Attacking drone velocity direction x component
Att_vel(:,2)=v.*sin(theta);    %Attacking drone velocity direction y component
timer=0;

Att_alive=ones(N_attacker,1);
Attacker_pos=5*rand([N_attacker,2]);
Defender_pos=40+5*rand([N_defender,2]);

% This preallocates the distance matrix for follow on loops.

Dist=zeros(N_defender,N_attacker);

Def_v=zeros(N_defender,2);
Def_Velocity_vect=zeros(N_defender,2);
Def_Acceleration=zeros(N_defender,2);

target_num=NaN*ones(N_defender,1);
totalkilled=0;

A = [];

while sum(Att_alive)>final_fraction*N_attacker
    iter=1;
    while iter<=N_defender
        iter2=1;
        while iter2<=N_attacker
            Dist(iter,iter2)=norm([Defender_pos(iter,1) Defender_pos(iter,2)]-[Attacker_pos(iter2,1) Attacker_pos(iter2,2)]);
            iter2=iter2+1;
        end
        iter=iter+1;
    end

    iter2=1;
    while iter2<=N_attacker
        Disto(1,iter2)=norm([0 0]-[Attacker_pos(iter2,1) Attacker_pos(iter2,2)]);
        iter2=iter2+1;
    end

    iter=1;

    while iter <=N_defender
        %destroy attacker within minimum range
        if ~isnan(target_num(iter,1))

            if(Dist(iter,target_num(iter,1))) <=kill_range
                iter2=target_num(iter,1);
                Attacker_pos(iter2,1)=NaN;
                Attacker_pos(iter2,2)=NaN;
                Dist(:,iter2)=NaN;
                Dist_check(:,iter2)=NaN;
                target_num(iter,1)=NaN;
                Att_Acceleration(iter2,1)=NaN;
                Att_Acceleration(iter2,2)=NaN;
                Att_vel(iter2,1)=0;
                Att_vel(iter2,2)=0;
                Att_alive(iter2,1)=0;
                totalkilled=totalkilled+1;
            else
            end

        else
        end
        iter=iter+1;
    end

    iteri=1;
    Dist_check=Dist;
    Disto_check=Disto;
    target_num=NaN*ones(N_defender,1);

    while iteri<=N_defender && iteri+totalkilled<=N_attacker
        maxMatrix=max(Disto_check(:));
        [iter3,iter4] = find(Disto_check==maxMatrix);
        minMatrix = min(Dist_check(:,iter4));
        %determine closest attacker/defender combos. exclude attacker
        %once already paired
        [iter,iter2] = find(Dist_check==minMatrix);
        Dist_check(iter,:)=NaN;
        Dist_check(:,iter2)=NaN;
        Disto_check(iter4)=NaN;
        target_num(iter,1)=iter2;

        if Att_alive(iter2,1)==1
            %calculate intercept and point acceleration vector to this
            %intercept
            xdiff=Attacker_pos(iter2,1)-Defender_pos(iter,1);
            ydiff=Attacker_pos(iter2,2)-Defender_pos(iter,2);
            c2=Att_vel(iter2,1)^2+Att_vel(iter2,2)^2-Defender_v_max^2;
            c3=2*xdiff*Att_vel(iter2,1)+2*ydiff*Att_vel(iter2,2);
            c4=xdiff^2+ydiff^2;
            ts=roots([c2 c3 c4]);
            ts=max(ts);

            Def_Velocity_vect(iter,1)=((xdiff+Att_vel(iter2,1)*ts))/ts;
            Def_Velocity_vect(iter,2)=((ydiff+Att_vel(iter2,2)*ts))/ts;

            %%
            vec=[Def_Velocity_vect(iter,1) Def_Velocity_vect(iter,2)];

            %%
            Def_Acceleration(iter,1)=Def_a*Def_Velocity_vect(iter,1)/norm(vec);
            Def_Acceleration(iter,2)=Def_a*Def_Velocity_vect(iter,2)/norm(vec);
        else
        end

        iteri=iteri+1;
    end
    Dist_check=Dist;

    iter=1;
    while iter<=N_attacker
        iter2=1;
        if Att_alive(iter,1)==1
            while iter2<=N_attacker

                %%
                % If the mth defender is equal to the nth attacker or the nth attacker is dead,
                % then the distance between the
                %
                % mth defender and nth attacker is excluded.

                if iter == iter2 ||  Att_alive(iter2,1)==0
                    Dist_att(iter,iter2)=NaN;
                    %%
                    % If the mth defender is different to the nth attacker or the nth attacker isn't
                    % dead, then the distance between
                    %
                    %

                else
                    Dist_att(iter,iter2)=norm([Attacker_pos(iter,1) Attacker_pos(iter,2)]-[Attacker_pos(iter2,1) Attacker_pos(iter2,2)]);
                end
                iter2=iter2+1;
            end
        else
            Dist_att(iter,:)=NaN;
        end
        iter=iter+1;
    end

    iteri=1;
    while iteri<=N_defender
        %pair multiple defenders to one attacker once attackers<def
        if target_num(iteri,1)==0 || isnan(target_num(iteri,1))
            [iter,iter2] = min(Dist_check(iteri,:));
            target_num(iteri,1)=iter2;
            if Att_alive(iter2,1)==1
                xdiff=Attacker_pos(iter2,1)-Defender_pos(iteri,1);
                ydiff=Attacker_pos(iter2,2)-Defender_pos(iteri,2);
                c2=Att_vel(iter2,1)^2+Att_vel(iter2,2)^2-Defender_v_max^2;
                c3=2*xdiff*Att_vel(iter2,1)+2*ydiff*Att_vel(iter2,2);
                c4=xdiff^2+ydiff^2;
                ts=roots([c2 c3 c4]);
                ts=max(ts);
                Def_Velocity_vect(iteri,1)=((xdiff+Att_vel(iter2,1)*ts))/ts;
                Def_Velocity_vect(iteri,2)=((ydiff+Att_vel(iter2,2)*ts))/ts;
                vec=[Def_Velocity_vect(iteri,1) Def_Velocity_vect(iteri,2)];
                Def_Acceleration(iteri,1)=Def_a*Def_Velocity_vect(iteri,1)/norm(vec);
                Def_Acceleration(iteri,2)=Def_a*Def_Velocity_vect(iteri,2)/norm(vec);
            else
            end
        else
        end
        iteri=iteri+1;
    end

    Heading_Angle=atan2(Def_Acceleration(:,2),Def_Acceleration(:,1));

    %update acc and v
    Def_Acceleration(:,1)=Def_Acceleration(:,1)-Def_v(:,1)*ramp_time;
    Def_Acceleration(:,2)=Def_Acceleration(:,2)-Def_v(:,2)*ramp_time;
    Def_v(:,1)=Def_v(:,1)+Def_Acceleration(:,1);
    Def_v(:,2)=Def_v(:,2)+Def_Acceleration(:,2);
    Attacker_pos(:,1)=Attacker_pos(:,1)+Att_vel(:,1);
    Attacker_pos(:,2)=Attacker_pos(:,2)+Att_vel(:,2);
    Attacker_pos_mag=norm([Attacker_pos(:,1) Attacker_pos(:,2)]);
    Defender_pos(:,1)=Defender_pos(:,1)+Def_v(:,1);
    Defender_pos(:,2)=Defender_pos(:,2)+Def_v(:,2);
    %         Heading_Angle=atan2(Def_Acceleration(:,2),Def_Acceleration(:,1));
    timer=timer+1;
    %plot
    %         if do_plot==1
    %             plot(Attacker_pos(:,1),Attacker_pos(:,2),'r.','MarkerSize',16)
    %             hold on;
    %             plot(Defender_pos(:,1),Defender_pos(:,2),'b.','MarkerSize',16)
    %             xlim([0 50])
    %             ylim([0 50])
    %             set(gca,'XTickLabel',[], 'YTickLabel', [])
    %             pause(.1);
    %             hold off;
    %         else
    %         end

    %start time once first attacker destroyed
    if sum(Att_alive)<N_attacker
        t=t+1;
    else
    end

    %         writematrix([timer' Attacker_pos(:,1)' Attacker_pos(:,2)' Defender_pos(:,1)' Defender_pos(:,2)' ...
    %         Att_vel(:,1)' Att_vel(:,2)' Def_v(:,1)' Def_v(:,2)',Heading_Angle'],'100sim.txt','WriteMode','append')

    %         writematrix([timer' Attacker_pos(:,1)' Attacker_pos(:,2)' Att_vel(:,1)' Att_vel(:,2)' Defender_pos(:,1)' Defender_pos(:,2)' ...
    %         Def_v(:,1)' Def_v(:,2)', Def_Acceleration(:,1)', Def_Acceleration(:,2)'],'IKsim.txt','WriteMode','append')

    A = [A; timer' Attacker_pos(:,1)' Attacker_pos(:,2)' Att_vel(:,1)' Att_vel(:,2)' Defender_pos(:,1)' Defender_pos(:,2)' ...
        Def_v(:,1)' Def_v(:,2)', Def_Acceleration(:,1)', Def_Acceleration(:,2)'];

end
%     save ('dims', 'N_attacker','N_defender')


% this file reads data from the 1st encounter produced in 100sim.txt.
% It then samples the resulting data to construct a
% Bernstein polynomial approiximation to be used for the initial guess in
% the optimization

%     close all;
%
%     clear all;
%     load dims;
data.eps = attacker_prob_survival; % attacker ma survival probability

%     data.ode_func = @new_ode4_Leon_withZombies;
%     %potential: followers to followers
%     data.SWARM.alpha_i = .5;
%     data.SWARM.d0 = 1;
%     data.SWARM.d1 = 6;
%
% gain of dissipative force in control law
%     data.SWARM.K_hvu = 5;
%     data.SWARM.K = 5;
%     data.SWARM.umax = 10;
%
%     %potential: followers to intruders
%     data.SWARM.alphaINT_i = 6;%alpha(i);%6.5; %here is where in this code alpha is set as uncertain parameter
%     data.SWARM.INTd0 = 3;
%     data.SWARM.INTd1 = data.SWARM.INTd0;
%     data.ATTACKERWEAPON.F = 0; data.ATTACKERWEAPON.a = 1;
%     data.ATTACKERWEAPON.lambda = .05;
%     data.ATTACKERWEAPON.sigma = 2;
%     data.wt = 0;
%     data.wp = 1;
%     data.DEFENDER.dmin = 0.01;
%     data.DEFENDER.Nx = 3; % 3D double integrator
%     data.DEFENDER.p_hvu = [0;0;0];


data.DEFENDER.vmax = 1;
data.DEFENDER.vmin = 0;
data.DEFENDER.vFlag = vflag; % active speed constraint
data.DEFENDER.umax = 1;

N = 15; % Bernsein Polynomial order

data.DEFENDERWEAPON.F = 0;
data.DEFENDERWEAPON.a = 1;
data.DEFENDERWEAPON.lambda = rate_of_fire_def;
data.DEFENDERWEAPON.sigma = range_def;


% A = readmatrix('IKsim.txt');

A = fill_nans(A);
K = size(A,1);

AttackersPos_x = A(:,2:N_attacker+1);
AttackersPos_y = A(:,N_attacker+2:2*N_attacker+1);

AttackersPos_x = AttackersPos_x(1:K-1,:);
AttackersPos_y = AttackersPos_y(1:K-1,:);

AttackersVel_x = A(1:K-1,2*N_attacker+2:3*N_attacker+1);
AttackersVel_y = A(1:K-1,3*N_attacker+2:4*N_attacker+1);

pointer = 4*N_attacker + 2; % defender data

DefendersPos_x = A(1:K-1,pointer:pointer + N_defender-1);
DefendersPos_y = A(1:K-1,pointer + N_defender:pointer + 2*N_defender-1);

DefendersVel_x = A(1:K-1,pointer + 2*N_defender:pointer + 3*N_defender-1);
DefendersVel_y = A(1:K-1,pointer + 3*N_defender:pointer + 4*N_defender-1);

DefendersAcc_x = A(1:K-1,pointer + 4*N_defender:pointer + 5*N_defender-1);
DefendersAcc_y = A(1:K-1,pointer + 5*N_defender:pointer + 6*N_defender-1);
DefendersAcc_x(1,:) = zeros(1,N_defender);
DefendersAcc_y(1,:) = zeros(1,N_defender);

% below I tried least squares to find control points. But it doesnt work
% well for velocity and acceleration

tf = computeTotalTime([DefendersPos_x(:,1) DefendersPos_y(:,1)], [DefendersVel_x(:,1) DefendersVel_y(:,1)]);
time = linspace(0,1,K-1);
tf_Nate = tf;
BN = bernsteinMatrix(N,time);
Dm = Diff_elev(N,1)';

Cx = BN\DefendersPos_x;
Cy = BN\DefendersPos_y;

P_x = BN*Cx;
P_y = BN*Cy;

V_x = BN*Dm*Cx/tf;
V_y = BN*Dm*Cy/tf;

% Acc_x = BN*Dm*Dm*Cx/tf^2;
% Acc_y = BN*Dm*Dm*Cy/tf^2;

if plotit == 1
    figure(14);  hold;
    for i = 1:N_attacker
        for j = 1:N_defender
            plot(P_x(:,j), P_y(:,j),'x');
            plot(P_x(:,j), P_y(:,j),'r');
            plot(AttackersPos_x(:,i), AttackersPos_y(:,i),'o');
        end
    end
    hold off
end


data.K = K;
data.N = N;
data.N_defender = N_defender;
data.N_attacker = N_attacker;
data.tf = tf;
data.time = time;

data.BN = BN;
data.Dm = Dm;

data.DefendersPos_x = DefendersPos_x;
data.DefendersPos_y = DefendersPos_y;

data.DefendersVel_x = DefendersVel_x;
data.DefendersVel_y = DefendersVel_y;

data.DefendersAcc_x = DefendersAcc_x;
data.DefendersAcc_y = DefendersAcc_y;

data.AttackersPos_x = AttackersPos_x;
data.AttackersPos_y = AttackersPos_y;

data.AttackersVel_x = AttackersVel_x;
data.AttackersVel_y = AttackersVel_y;

Cx = Cx(2:end,:);
Cy = Cy(2:end,:);
Cx0 = DefendersPos_x(1,:);
Cy0 = DefendersPos_y(1,:);
data.Cx0 = Cx0;
data.Cy0 = Cy0;

x1 = reshape(Cx,N_defender*(N),1);
x2 = reshape(Cy,N_defender*(N),1);
%
% x1 = reshape(Cx,N_defender*(N+1),1);
% x2 = reshape(Cy,N_defender*(N+1),1);

x0 = [x1;x2;tf];

Qmax_Nate = new_ode4_Leon_withZombies_v3(x0,data);

c = @(x)myconstraints(x,data);
% costFunc = @(x)new_ode4_Leon_withZombies(x,data);
cost = @(x)costFunc(x,data);
A =[];b = [];
Aeq = [];beq = [];
M = length(x0) - 1;

% make sure tf is positive
lb = [-inf*ones(M,1);0]; ub = [inf*ones(M,1);tf];

options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'MaxFunctionEvaluations',1e6,...
    'ConstraintTolerance',1e-4,...
    'StepTolerance',1e-4,...
    'MaxIterations',100);
%                        "EnableFeasibilityMode",true,...
%                        'Display','iter','OptimalityTolerance',1e-8,...
% %                       'FunctionTolerance',1e-8,..
%                       'ConstraintTolerance',1e-3);
tic
[x_opt,Jout,exitflag,output] = fmincon(cost,x0,A,b,Aeq,beq,lb,ub,c,options);
toc
disp(['Final Cost, J = ' num2str(Jout)]);

% options = optimoptions(@fminunc,...
%                        'MaxFunctionEvaluations',1e6,...
%                        'StepTolerance',1e-4,...
%                        'Display','iter','MaxIterations',60);
% %                        'OptimalityTolerance',1e-8,...
% % %                       'FunctionTolerance',1e-8,..
% %                       'ConstraintTolerance',1e-3);
%
% [x_opt,Jout,exitflag,output]  = fminunc(costFunc,x0);
% disp(['Final Cost, J = ' num2str(Jout)]);

x = x_opt;
[Qmax_optimal,max_def_att_rate] = new_ode4_Leon_withZombies_v3_add_effdef(x,data);
tf = x(end);
tf_optimal = tf;

x1 = x(1:(N)*N_defender);
x2 = x((N)*N_defender+1:end-1);

Cx = reshape(x1,N,N_defender);
Cy = reshape(x2,N,N_defender);

Cx = [Cx0;Cx];
Cy = [Cy0;Cy];

Pos_x = BN*Cx;
Pos_y = BN*Cy;

Vel_x = BN*Dm*Cx/tf;
Vel_y = BN*Dm*Cy/tf;

Acc_x = BN*Dm*Dm*Cx/tf^2;
Acc_y = BN*Dm*Dm*Cy/tf^2;

if plotit == 1
    figure(24);  hold;
    for i = 1:N_attacker
        for j = 1:N_defender
            plot(Pos_x(:,j), Pos_y(:,j),'x');
            plot(Pos_x(:,j), Pos_y(:,j),'r');
            plot(AttackersPos_x(:,i), AttackersPos_y(:,i),'o');
        end
    end
    hold off

    figure(36);
    plot(time,Vel_y); hold;
    plot(time,DefendersVel_y,'o'); hold off

    figure(37);
    plot(time,Acc_x); hold;
    plot(time,DefendersAcc_x,'o'); hold off
end
distance_2defenders = pdist2([AttackersPos_x(end,:);AttackersPos_y(end,:)]', [Pos_x(end,:);Pos_y(end,:)]');
[i,j] = find(distance_2defenders <= 1.5*range_def);
j = unique(j);
% effective_defenders = length(j);
effective_defenders = sum(max_def_att_rate>1e-3);

if ~plotit
    save(file_name,'Qmax_optimal','Qmax_Nate','tf_optimal','tf_Nate','effective_defenders')
end


end
%% cost function
function J = costFunc(x,data)

J = x(end);

end
%% constraint function
function [C,Ceq] = myconstraints(x,data)

N = data.N;

tf = x(end);
BN = data.BN;
Dm = data.Dm;
umax = data.DEFENDER.umax;

x1 = x(1:(N)*data.N_defender);
x2 = x((N)*data.N_defender+1:end-1);

Cx = reshape(x1,N,data.N_defender);
Cy = reshape(x2,N,data.N_defender);

Cx = [data.Cx0;Cx];
Cy = [data.Cy0;Cy];

% Pos_x = BN*Cx;
% Pos_y = BN*Cy;
%
Vel_x = BN*Dm*Cx/tf;
Vel_y = BN*Dm*Cy/tf;
Vel_vecx = reshape(Vel_x.*Vel_x,(data.K-1)*data.N_defender,1);
Vel_vecy = reshape(Vel_y.*Vel_y,(data.K-1)*data.N_defender,1);

Acc_x = BN*Dm*Dm*Cx/tf^2;
Acc_y = BN*Dm*Dm*Cy/tf^2;

% Acc_x = Dm*Dm*Cx/tf^2;
% Acc_y = Dm*Dm*Cy/tf^2;

% infinity norm bound
max_Accx = (Acc_x.*Acc_x);
max_Accy = (Acc_y.*Acc_y);

max_Accx = reshape(max_Accx,(data.K-1)*data.N_defender,1);
max_Accy = reshape(max_Accy,(data.K-1)*data.N_defender,1);

% max_Accx = reshape(max_Accx,1,(data.N+1)*data.N_defender);
% max_Accy = reshape(max_Accy,1,(data.N+1)*data.N_defender);

if data.DEFENDER.vFlag == 0
    C = [max_Accx - umax^2; max_Accy - umax^2;...
        Vel_vecx - data.DEFENDER.vmax^2; ...
        Vel_vecy - data.DEFENDER.vmax^2];
else
    C = [max_Accx - umax^2; max_Accy - umax^2;...
        Vel_vecx - data.DEFENDER.vmax^2; ...
        Vel_vecy - data.DEFENDER.vmax^2; ...
        -Vel_vecy + data.DEFENDER.vmin^2; ...
        -Vel_vecx + data.DEFENDER.vmin^2
        ];
end

Qavg = new_ode4_Leon_withZombies_v3(x,data);
% maxspeed = (VN.*VN);
C = [C;Qavg - data.eps];
Ceq = [];

end

function A = fill_nans(A)
% Replaces the nans in each column with
% previous non-nan values.
for ii = 1:size(A,2)
    I = A(1,ii);
    for jj = 2:size(A,1)
        if isnan(A(jj,ii))
            A(jj,ii) = I;
        else
            I  = A(jj,ii);
        end
    end
end
end
function total_time = computeTotalTime(position, velocity)
% position and velocity should be Nx2 matrices, where N is the number of time steps

% Calculate the displacement between consecutive positions
displacement = [0 0;diff(position)];

% Calculate the distance traveled at each time step
distance = hypot(displacement(:,1), displacement(:,2));

% Calculate the time interval between consecutive time steps
%     time_interval = diff(time);

% Calculate the velocity magnitude at each time step
velocity_mag = hypot(velocity(:,1), velocity(:,2));

% Calculate the total time traveled
total_time = sum(distance./ velocity_mag);
end


%% Bezier derivative
function Dm = Diff(N,tf )
% derivative of a Bezier curve
% INPUT
% N: number of nodes
% OUTPUT
% Dm{N}: differentiation matrix for bez curves of order N (N+1 ctrl points)

% Notes:
% If Cp are the control points of bez, then the control points of bezdot are Cpdot = Cp*Dm
% To compute bezdot with N ctrl points, degree elevation must be performed

Dm = -[N/tf*eye(N); zeros(1,N)]+[zeros(1,N);N/tf*eye(N)];

end

function Telev = deg_elev(N)
% INPUT
% N order of Bezier curve
% OUTPUT
% Telev{N}: Transformation matrix from Nth order (N+1 control points) to
% (N+1)th order (N+2 control points)
% If Cp is of order N-1, then Cp*Telev{N-1} is of order N
% see Equation (12+1) in https://pdfs.semanticscholar.org/f4a2/b9def119bd9524d3e8ebd3c52865806dab5a.pdf
% Paper: A simple matrix form for degree reduction of Be´zier curves using ChebyshevBernstein basis transformations

if N < 5
    es='ERROR: The approximation order should be at least 5';
    disp(es); Dm = [];
    return
end


for i = 1:1:N
    Telev{i} = zeros(i+2,i+1);
    for j = 1:1:i+1
        Telev{i}(j,j) = i+1-(j-1);
        Telev{i}(j+1,j) = 1+(j-1);
    end
    Telev{i} = 1/(i+1)*Telev{i}';
end

end




function Dm = Diff_elev(N,tf)
% derivative of a Bezier curve
% INPUT
% N: number of nodes, tf: final time
% OUTPUT
% Dm{N}: differentiation matrix for bez curves of order N (N+1 ctrl points)
% The differentiation matrix is (N+1)by(N+1), ie. differently from Diff,
% this matrix gives a derivative of the same order of the curve to be
% differentiated

% Notes:
% If Cp are the control points of bez, then the control points of bezdot are Cpdot = Cp*Dm


Dm = Diff(N,tf);
Telev = deg_elev(N);
Dm = Dm*Telev{N-1};

end


%% Generate Product Matrix
function Prod_T = Prod_Matrix(N)
%This function produces a matrix which can be used to compute ||x dot x||^2
% i.e. xaug = x'*x;
% xaug = reshape(xaug',[length(x)^2,1]);
% y = Prod_T*xaug;
% or simply norm_square(x)


T = zeros(2*N+1,(N+1)^2);

for j = 0:2*N
    for i = max(0,j-N): min(N,j)
        if N >= i && N >= j-i && 2*N >= j && j-i >= 0
            T(j+1,N*i+j+1) = nchoosek_mod(N,i)*nchoosek_mod(N,j-i)/nchoosek_mod(2*N,j);
        end
    end
end

Prod_T = T;


end


%% Generate n choose k_mod

function out = nchoosek_mod(n,k)

out = 1;
for i = 1:k
    out = out*(n-(k-i));
    out = out/i;
end
end

function yout = new_ode4_Leon_withZombies_v3(x,data)

N = data.N;

N_attacker = data.N_attacker;
N_defender = data.N_defender;
tf = data.tf;
time = data.time*tf;
dt = time(2) - time(1);
BN = data.BN;

Cx0 = data.Cx0;
Cy0 = data.Cy0;

AttackersPos_x = data.AttackersPos_x;
AttackersPos_y = data.AttackersPos_y;

% AttackersVel_x = data.AttackersVel_x;
% AttackersVel_y = data.AttackersVel_y;

% x1 = x(1:(N+1)*N_defender);
% x2 = x((N+1)*N_defender+1:end);
%
% Cx = reshape(x1,N+1,N_defender);
% Cy = reshape(x2,N+1,N_defender);
%
% x1 = x(1:(N+1)*N_defender);
% x2 = x((N+1)*N_defender+1:end);

x1 = x(1:(N)*N_defender);
x2 = x((N)*N_defender+1:end-1);

Cx = reshape(x1,N,N_defender);
Cy = reshape(x2,N,N_defender);

Cx = [Cx0;Cx];
Cy = [Cy0;Cy];

Pos_x = BN*Cx;
Pos_y = BN*Cy;

Q = ones(N_attacker,1); %probability of attacker survival
Pd = ones(N_defender,1); %prob of def surv
Pdmat = repmat(Pd',[N_attacker 1]);

M = length(time);
for t = 1:M
    x_att = AttackersPos_x(t,:);
    y_att = AttackersPos_y(t,:);

    % c. Defender forces computation
    x_def = Pos_x(t,:);
    y_def = Pos_y(t,:);

    distance_2defenders = pdist2([x_att;y_att]', [x_def;y_def]');
    %  compute probability of attacker/defender/hvu survival
    rda2d = data.DEFENDERWEAPON.lambda*normcdf((data.DEFENDERWEAPON.F - data.DEFENDERWEAPON.a*distance_2defenders.^2)/data.DEFENDERWEAPON.sigma,0,1);
    Q = Q.*(1-(1-prod(1-(rda2d.*Pdmat)'.*dt)))';

end
yout = max(Q);

end

function [yout,max_def_att_rate] = new_ode4_Leon_withZombies_v3_add_effdef(x,data)

N = data.N;

N_attacker = data.N_attacker;
N_defender = data.N_defender;
tf = data.tf;
time = data.time*tf;
dt = time(2) - time(1);
BN = data.BN;

Cx0 = data.Cx0;
Cy0 = data.Cy0;

AttackersPos_x = data.AttackersPos_x;
AttackersPos_y = data.AttackersPos_y;

% AttackersVel_x = data.AttackersVel_x;
% AttackersVel_y = data.AttackersVel_y;

% x1 = x(1:(N+1)*N_defender);
% x2 = x((N+1)*N_defender+1:end);
%
% Cx = reshape(x1,N+1,N_defender);
% Cy = reshape(x2,N+1,N_defender);
%
% x1 = x(1:(N+1)*N_defender);
% x2 = x((N+1)*N_defender+1:end);

x1 = x(1:(N)*N_defender);
x2 = x((N)*N_defender+1:end-1);

Cx = reshape(x1,N,N_defender);
Cy = reshape(x2,N,N_defender);

Cx = [Cx0;Cx];
Cy = [Cy0;Cy];

Pos_x = BN*Cx;
Pos_y = BN*Cy;

Q = ones(N_attacker,1); %probability of attacker survival
Pd = ones(N_defender,1); %prob of def surv
Pdmat = repmat(Pd',[N_attacker 1]);

max_def_att_rate = zeros(1,N_defender);

M = length(time);
for t = 1:M
    x_att = AttackersPos_x(t,:);
    y_att = AttackersPos_y(t,:);

    % c. Defender forces computation
    x_def = Pos_x(t,:);
    y_def = Pos_y(t,:);

    distance_2defenders = pdist2([x_att;y_att]', [x_def;y_def]');
    %  compute probability of attacker/defender/hvu survival
    rda2d = data.DEFENDERWEAPON.lambda*normcdf((data.DEFENDERWEAPON.F - data.DEFENDERWEAPON.a*distance_2defenders.^2)/data.DEFENDERWEAPON.sigma,0,1);
    Q = Q.*(1-(1-prod(1-(rda2d.*Pdmat)'.*dt)))';
    curr_max_att_rate = sum(rda2d);
    max_def_att_rate(curr_max_att_rate>max_def_att_rate)=curr_max_att_rate(curr_max_att_rate>max_def_att_rate);

end
yout = max(Q);

end

