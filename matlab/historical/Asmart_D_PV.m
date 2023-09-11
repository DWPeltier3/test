function [states] = Asmart_D_PV(attacker,defender,defender_v,do_plot,kill_pro,seed,accel,kill_range)
    
    close all;
    rng(seed); %specifies seed for random number generator
    t=0;
    N_def=attacker; %# attackers
    N_att=defender;
    Att_v=defender_v; %defender velocity (constant)
    steps_to_accel=accel;
    ramp_time=1/steps_to_accel;
    Att_a=Att_v*ramp_time; %defender velocity increment
    
    vel_min=.05; vel_max=0.4; vm = vel_max-vel_min;
    v = vel_min+vm.*rand(N_def,1); %(Nx1)column vector of attacker velocities (constant)
    theta = pi/2.*rand(N_def,1); %(Nx1)col vector of attacker heading (constant)
    vel(:,1)=v.*cos(theta); %(Nx2)attacker x&y velocities (constant)
    vel(:,2)=v.*sin(theta);
    
    Def_alive=ones(N_def,1);        % attacker alive (=1) col vector
    Def_pos=5*rand([N_def,2]);      % attacker initial position (x,y)=[0,5]
    Att_pos=40+5*rand([N_att,2]);   % defender initial position
    a_vel=zeros(N_att,2);           % defender initial velocity
    
    % file_name=['results_' num2str(attacker) '_' num2str(defender) '_' num2str(seed) '_' num2str(kill_pro) '_' num2str(ramp_time) '.mat'];
    
    %% Prepare data to be saved for  NN training
    states=[Att_pos a_vel]; % initial state matrix: row=defender ONLY; col=states (PVA):PxPyVxVy
    % Flatten state vector into pages: features along 3rd dimension; column=timestep; row=sample (seed;run)
    states=reshape(states,1,1,[]); % # pages = # agents * # features

    %% RUN SIMULATION
    while sum(Def_alive)>kill_pro*N_def %while #att alive > #def*constant

        Dist=zeros(N_att,N_def); %distance matrix (row=defender, col=attacker)
        iter=1; %init counter

        while iter<=N_att %calculate distance between every attacker and defender
            iter2=1;
            while iter2<=N_def
                Dist(iter,iter2)=norm([Att_pos(iter,1) Att_pos(iter,2)]-[Def_pos(iter2,1) Def_pos(iter2,2)]);
                iter2=iter2+1;
            end
           iter=iter+1;
        end

        iter=1;
        target_num=zeros(N_att,1); %col vector = # def; defender->attacker assignments
        Dist_check=Dist; %matrix (row=defender, col=attacker)

        while iter<=N_att %for each defender
            minMatrix = min(Dist_check(:)); %find closest attacker
            [row,col] = find(Dist_check==minMatrix); %match defender to closest attacker
            Dist_check(row,:)=NaN; %prevent multiple defenders matched to same attacker
            Dist_check(:,col)=NaN;
            if(min(Dist(row,:))) <kill_range %can defender kill closest attacker
                Def_pos(col,1)=NaN;
                Def_pos(col,2)=NaN;
                vel(col,1)=0;
                vel(col,2)=0;
                Def_alive(col,1)=0;
            end
            if Def_alive(col,1)==1 %if can kill closest attacker, move towards it
                xdiff=Def_pos(col,1)-Att_pos(row,1);
                ydiff=Def_pos(col,2)-Att_pos(row,2);
                vec=[xdiff ydiff];
                avel(row,1)=Att_a*vec(1)/norm(vec);
                avel(row,2)=Att_a*vec(2)/norm(vec);
            end
            target_num(row,1)=col;
            iter=iter+1;
        end

        iter=1; %defender
        while iter<=N_att %for each defender
            if target_num(iter,1)==0 %defender not assigned an attacker
                [~,I] = min(Dist(iter,:)); %find closest attacker (index) to defender
                if(min(Dist(iter,:))) <kill_range %can defender kill closest attackers?
                    Def_pos(I,1)=NaN; %kill closest attacker
                    Def_pos(I,2)=NaN;
                    vel(I,1)=0;
                    vel(I,2)=0;
                    Def_alive(I,1)=0;
                end
                if Def_alive(I,1)==1 %if can't kill closest attacker alive (what if could kill closest: could sort attackers close->far, then check next closest
                    xdiff=Def_pos(I,1)-Att_pos(iter,1); %x&y dist to closest attacker
                    ydiff=Def_pos(I,2)-Att_pos(iter,2);
                    vec=[xdiff ydiff];
                    avel(iter,1)=Att_a*vec(1,1)/norm(vec); %defender xvelocity towards attacker (unit vector)
                    avel(iter,2)=Att_a*vec(1,2)/norm(vec); %defender yvelocity towards attacker (unit vector)
                end
                target_num(iter,1)=I; %assign closest alive attacker to defender
            end
            iter=iter+1;
        end %repeat for all defenders (assign attacker)

        avel(:,1)=avel(:,1)-a_vel(:,1)*ramp_time;   % defender acceleration x
        avel(:,2)=avel(:,2)-a_vel(:,2)*ramp_time;   % defender acceleration y
        a_vel(:,1)=a_vel(:,1)+avel(:,1);            % defender velocity x
        a_vel(:,2)=a_vel(:,2)+avel(:,2);            % defender velocity y
        Def_pos(:,1)=Def_pos(:,1)+vel(:,1);         % update attacker x pos=xprev+xvel
        Def_pos(:,2)=Def_pos(:,2)+vel(:,2);         % update attacker y pos=yprev+yvel
        Att_pos(:,1)=Att_pos(:,1)+a_vel(:,1);       % update defender x pos=xprev+xvel
        Att_pos(:,2)=Att_pos(:,2)+a_vel(:,2);       % update defender y pos=xprev+xvel
        
        if do_plot==1
            %switched Def 'r' and Att 'b' colors for presentation
            plot(Def_pos(:,1),Def_pos(:,2),'b.','MarkerSize',16)
            hold on;
            xlim([0 50])
            ylim([0 50])
            plot(Att_pos(:,1),Att_pos(:,2),'r.','MarkerSize',16)
            pause(.1);
            hold off;
        end

        %start time once first attacker destroyed
        if sum(Def_alive)<N_def
          t=t+1;
        end

        %Update 'states' matrix history for output
        newstate=[Att_pos a_vel]; % defender only
        newstate=reshape(newstate,1,1,[]);
        states=cat(2,states,newstate); %add new column (time step) with pages (updated states)
    end
    % save(file_name,'t')
end

