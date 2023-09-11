function [states] = Agreedy_D_PV(attacker, defender, defender_v, do_plot, kill_pro, seed)
    
    close all;
    t=0;
    rng(seed); %specify seed for random number generator
    N_def=attacker;
    N_att=defender;
    Att_v=defender_v; %defender velocity
    
    % initialize attacker velocities
    vel_min=.05;
    vel_max=0.4;
    vm = vel_max-vel_min;
    v = vel_min+vm.*rand(N_def,1);
    theta = pi/2.*rand(N_def,1);
    vel(:,1)=v.*cos(theta); %attacker Vx
    vel(:,2)=v.*sin(theta); %attacker Vy
    
    Def_alive=ones(N_def,1);
    Def_pos=5*rand([N_def,2]);
    Att_pos=40+5*rand([N_att,2]);
    
    %% Prepare data to be saved for  NN training
    states=[Att_pos zeros(N_att,2)]; %initial state matrix: rows=defender ONLY; col=states (PV):PxPyVxVy
    % Flatten state vector into pages: features (PV) along 3rd dimension (pages); column=timestep; row=sample (seed;run)
    states=reshape(states,1,1,[]); % # pages = # agents * # features
    
    %% RUN SIMULATION
    while sum(Def_alive)>kill_pro*N_def
        Dist=zeros(N_att,N_def);
        iter=1;
        while iter<=N_att
            iter2=1;
            while iter2<=N_def
                Dist(iter,iter2)=norm([Att_pos(iter,1) Att_pos(iter,2)]-[Def_pos(iter2,1) Def_pos(iter2,2)]);
                iter2=iter2+1;
            end
           iter=iter+1; 
        end

        iter=1;
        target_num=zeros(N_att,1);
        while iter<=N_att %for each defender
            [~,I] = min(Dist(iter,:));
            if(min(Dist(iter,:))) <.6 %kill range?
                Def_pos(I,1)=NaN;
                Def_pos(I,2)=NaN;
                Dist(:,I)=NaN;
                target_num(iter,1)=NaN;
                vel(I,1)=0;
                vel(I,2)=0;
                Def_alive(I,1)=0;
            end
            if Def_alive(I,1)==1
                xdiff=Def_pos(I,1)-Att_pos(iter,1);
                ydiff=Def_pos(I,2)-Att_pos(iter,2);
                vec=[xdiff ydiff];
                avel(iter,1)=Att_v*vec(1,1)/norm(vec); %update defender Vx
                avel(iter,2)=Att_v*vec(1,2)/norm(vec); %update defender Vy
            end
            target_num(iter,1)=I;
            iter=iter+1;
        end

        %plot
        if do_plot==1
            plot(Def_pos(:,1),Def_pos(:,2),'r.','MarkerSize',16)
            hold on;
            plot(Att_pos(:,1),Att_pos(:,2),'b.','MarkerSize',16)
            xlim([0 50])
            ylim([0 50])
            set(gca,'XTickLabel',[], 'YTickLabel', [])
%             xlim([0 200])
%             ylim([0 200])
            pause(.1)
            hold off;
        end

        Def_pos(:,1)=Def_pos(:,1)+vel(:,1);
        Def_pos(:,2)=Def_pos(:,2)+vel(:,2);
        Att_pos(:,1)=Att_pos(:,1)+avel(:,1); %update defender Px =xprev+xvel
        Att_pos(:,2)=Att_pos(:,2)+avel(:,2); %update defender Py =xprev+xvel
        
        %start time once first attacker destroyed
        if sum(Def_alive)<N_def
          t=t+1;
        end

        %Update 'states' matrix history for output
        newstate=[Att_pos avel]; %defender ONLY
        newstate=reshape(newstate,1,1,[]);
        states=cat(2,states,newstate); %add new column (time step) with pages (updated states)

    end
end