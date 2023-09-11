%% Collect & Save Data
% Use this script to run "greedy", "smart", and "merge" using inputs
% specified below "seedrange" number of times and collect defender 
% (and optionally attacker) data (position, velocity)

%% Number of runs per class
seedrange=400; % # of samples (runs) to collect

%% Simulation Inputs
N_attacker=10;     % number of attackers
N_defender=7;     % number of defenders (agents that kill; "attackers" for ONR)
Defender_v_max=1;   % defender velocity maximum
do_plot=0;          % 1=plot ; 0=no plot
final_fraction=0;   % final proportion of attackers remaining vs. #defenders
accel=10;           % defender acceleration steps
kill_range=1;       % defender weapons range (kill attacker range

%% Save Data Flag and File Name
savemat=true; %"true" will save all runs into a MATLAB .mat file

%% Attack plan: uncomment attack plan for "Amerge_DA_PVA_plan"
% shape: ball, wall, spear
% path: middle
% delay: 0 to any number (# = iterations to delay start of group)
%{
g1=dictionary("shape",{'ball'},"path",{'middle'},"delay",{0})   % group 1
g2=dictionary("shape",{'ball'},"path",{'middle'},"delay",{30})  % group 2
g3=dictionary("shape",{'ball'},"path",{'middle'},"delay",{60})  % group 3
plan={g1;g2;g3}                                                 % combine groups into attack plan
%}

%% Run simulation multiple times and gather training data and labels
for seed=1:seedrange

    %% MERGE
    % sm=Amerge_DA_PVA_plan(plan,N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
    % sm=Amerge_DA_PVA(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
    % sm=Amerge_DA_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
    sm=Amerge_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);

    %% SMART
    % ss=Asmart_DA_PVA(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
    % ss=Asmart_DA_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);
    ss=Asmart_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range);

    %% GREEDY
    % sg=Agreedy_DA_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed);
    sg=Agreedy_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed);

    if seed==1 %initialize data matricies
        data_m={sm};
        data_s={ss};
        data_g={sg};
    else %append data into matricies
        data_m=cat(1,data_m,{sm});
        data_s=cat(1,data_s,{ss});
        data_g=cat(1,data_g,{sg});
    end
end


%% Save data matricies
if savemat
    save('data_m.mat','data_m')
    save('data_s.mat','data_s')
    save('data_g.mat','data_g')
end
%{%}