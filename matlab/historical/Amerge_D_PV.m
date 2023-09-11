function [states] = Amerge_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range)
    % Collects defender data

    %set initial parameters
    close all;
    t=0;
    rng(seed);

    %set defender acceleration ramp up
    steps_to_accel=accel;
    ramp_time=1/steps_to_accel;
    Def_a=Defender_v_max*ramp_time;
    
    %attacker velocity init (constant)
    Att_v_min=.05;
    Att_v_max=0.4;
    vm = Att_v_max-Att_v_min;
    v = Att_v_min+vm.*rand(N_attacker,1);
    theta = pi/2.*rand(N_attacker,1);
    Att_vel(:,1)=v.*cos(theta);
    Att_vel(:,2)=v.*sin(theta);

    timer=0;
    Att_alive=ones(N_attacker,1);
    Attacker_pos=5*rand([N_attacker,2]);
    Defender_pos=40+5*rand([N_defender,2]);
    Dist=zeros(N_defender,N_attacker);
    Def_v=zeros(N_defender,2);
    Def_Velocity_vect=zeros(N_defender,2);
    Def_Acceleration=zeros(N_defender,2);
    target_num=NaN*ones(N_defender,1); %col vector of NaN x Ndef
    totalkilled=0;
    
    %% Prepare data to be saved for  NN training
    states=[Defender_pos Def_v]; % initial state matrix: row=defender ONLY; col=states (PV):PxPyVxVy
    % a=3; % # of agents states to save
    % states=[Defender_pos(1:a,:) Def_v(1:a,:)];
    % Flatten state vector into pages: features along 3rd dimension; column=timestep; row=sample (seed;run)
    states=reshape(states,1,1,[]); % # pages = # agents * # features

    %% Save data from ONE RUN ONLY
    % savemat=false; %"true" will save ONE RUN (current run) only
    % file_name=[num2str(N_attacker) '_' num2str(N_defender) '_' num2str(kill_range) '_' num2str(final_fraction) '_' num2str(accel) '_' num2str(seed) '_' num2str(Defender_v_max) '.mat'];
    % file_name='test.mat';

    %% RUN SIMULATION
    while sum(Att_alive)>final_fraction*N_attacker
        
        %Distances between each defender and attacker
        iter=1;
        while iter<=N_defender
            iter2=1;
            while iter2<=N_attacker
                Dist(iter,iter2)=norm([Defender_pos(iter,1) Defender_pos(iter,2)]-[Attacker_pos(iter2,1) Attacker_pos(iter2,2)]);
                iter2=iter2+1;
            end
           iter=iter+1; 
        end

        %Attacker distance to origin (row vector); not really used?
        iter2=1;
        while iter2<=N_attacker
            Disto(1,iter2)=norm([0 0]-[Attacker_pos(iter2,1) Attacker_pos(iter2,2)]);
            iter2=iter2+1;
        end
        
        %What does this sectoin do?
        iter=1;
        while iter <=N_defender
            %destroy attacker within minimum range
            if ~isnan(target_num(iter,1)) %if NOT NaN
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
                end
            end
            iter=iter+1;
        end

        %What does this sectoin do?
        iteri=1;
        Dist_check=Dist;
        Disto_check=Disto; %row vector
        target_num=NaN*ones(N_defender,1);
        while iteri<=N_defender && iteri+totalkilled<=N_attacker
            maxMatrix=max(Disto_check(:)); %scalar (max attacker distance from origin)
            [~,iter4] = find(Disto_check==maxMatrix); %index of attacker furthest from origin
            minMatrix = min(Dist_check(:,iter4));
            %determine closest attacker/defender combos. exclude attacker
            %once already paired
            [iter,iter2] = find(Dist_check==minMatrix);
            Dist_check(iter,:)=NaN;
            Dist_check(:,iter2)=NaN;
            Disto_check(iter4)=NaN; %not used after this (waste? of variables)
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
                vec=[Def_Velocity_vect(iter,1) Def_Velocity_vect(iter,2)];
                Def_Acceleration(iter,1)=Def_a*Def_Velocity_vect(iter,1)/norm(vec);
                Def_Acceleration(iter,2)=Def_a*Def_Velocity_vect(iter,2)/norm(vec);
            end
            iteri=iteri+1;
        end
        
        %What does this section do?
        iter=1;
        while iter<=N_attacker
            iter2=1;
            if Att_alive(iter,1)==1 
                while iter2<=N_attacker
                    if iter == iter2 ||  Att_alive(iter2,1)==0
                        Dist_att(iter,iter2)=NaN;
                    else
                        Dist_att(iter,iter2)=norm([Attacker_pos(iter,1) Attacker_pos(iter,2)]-[Attacker_pos(iter2,1) Attacker_pos(iter2,2)]);
                    end
                    iter2=iter2+1;
                end
                Dist_att(iter,:)=NaN;
            end
            iter=iter+1; 
        end
        
        %pair multiple defenders to one attacker once attackers<def
        Dist_check=Dist;
        iteri=1;
        while iteri<=N_defender
            if target_num(iteri,1)==0 || isnan(target_num(iteri,1))
                [~,iter2] = min(Dist_check(iteri,:));
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
                end
            end
            iteri=iteri+1;
        end

        %update state vectors (position, velocity, acceleration)
        Def_Acceleration(:,1)=Def_Acceleration(:,1)-Def_v(:,1)*ramp_time;
        Def_Acceleration(:,2)=Def_Acceleration(:,2)-Def_v(:,2)*ramp_time;
        Def_v(:,1)=Def_v(:,1)+Def_Acceleration(:,1);
        Def_v(:,2)=Def_v(:,2)+Def_Acceleration(:,2);
        Attacker_pos(:,1)=Attacker_pos(:,1)+Att_vel(:,1);
        Attacker_pos(:,2)=Attacker_pos(:,2)+Att_vel(:,2);
        Defender_pos(:,1)=Defender_pos(:,1)+Def_v(:,1);
        Defender_pos(:,2)=Defender_pos(:,2)+Def_v(:,2);
        timer=timer+1;
        
        %plot
        if do_plot==1
            %switched att 'r' and def 'b' colors for presentation
            plot(Attacker_pos(:,1),Attacker_pos(:,2),'b.','MarkerSize',16)
            hold on;
            plot(Defender_pos(:,1),Defender_pos(:,2),'r.','MarkerSize',16)
            xlim([0 50])
            ylim([0 50])
            set(gca,'XTickLabel',[], 'YTickLabel', [])
            pause(.1);
            hold off;
        end

        %start time once first attacker destroyed
        if sum(Att_alive)<N_attacker
            t=t+1;
        else
        end

        %Update 'states' matrix history for output
        newstate=[Defender_pos Def_v]; %PV of ALL agents
        % newstate=[Defender_pos(1:a,:) Def_v(1:a,:)]; %PV of "a" # of agents
        newstate=reshape(newstate,1,1,[]);
        states=cat(2,states,newstate); %add new column (time step) with pages (updated states)

    end
    
    % used to save ONLY ONE RUN
    % if savemat
    %     save(strcat('states', file_name),'states');
    % end

end

