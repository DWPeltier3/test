%% Collect Data
% Use this script to run "Amerge_DA_PVA" variants using inputs
% specified below "seedrange" number of times and collect defender &/or 
% attacker data (position, velocity, acceleration)

clear; clc

%% Simulation Inputs
N_attacker=10;     % number of attackers
N_defender=10;     % number of defenders (agents that kill; "attackers" for ONR)
Defender_v_max=1;   % defender velocity maximum
do_plot=0;          % 1=plot ; 0=no plot
final_fraction=0;   % final proportion of attackers remaining vs. #defenders
accel=10;           % defender acceleration steps
kill_range=1;       % defender weapons range (kill attacker range)

%% Save Data Flag and File Name
savemat=true; %"true" will save all runs into a MATLAB .mat file
file_name='_m.mat';

%% Attack plan
% Uncomment attack plan for "Amerge_DA_PVA"

% shape: ball, wall, spear
% path: middle
% delay: 0 to any number (# = iterations to delay start of group)

%{
g1=dictionary("shape",{'ball'},"path",{'middle'},"delay",{0})   % group 1
g2=dictionary("shape",{'ball'},"path",{'middle'},"delay",{30})  % group 2
g3=dictionary("shape",{'ball'},"path",{'middle'},"delay",{60})  % group 3
plan={g1;g2;g3}                                                 % combine groups into attack plan
%}

%{
%% Single Run Only
seed=2;             %random generate seed
% states=Amerge_DA_PVA(plan,N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
s=Amerge_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
%}


%% Run simulation multiple times and gather training data and labels
seedrange=400; % # of samples (runs) to collect
for seed=1:seedrange
    % s=Amerge_DA_PVA(plan,N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
    % s=Amerge_DA_PVA_original(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
    s=Amerge_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
    if seed==1
        data={s};
    else
        data=cat(1,data,{s});
    end
end

%save label and data mat
if savemat
    save(strcat('data', file_name),'data')
end
%{%}