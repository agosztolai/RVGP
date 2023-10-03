function params = config_loop(params,s)

    tic

    params.current_subject = s;
    disp('------------------------------------------------------------------------');
    fprintf('%s \n', strcat('Subject ', num2str(s), '/',num2str(params.numSub),' ::Info:: ~ neural-flows ~'));       
    disp('------------------------------------------------------------------------');

end
