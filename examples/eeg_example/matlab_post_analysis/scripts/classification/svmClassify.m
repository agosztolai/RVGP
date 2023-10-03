function SVMResults = svmClassify(SVMClassify,params)

    disp(' ... Classifying AD vs healthy controls...');

    SVMResults = cell(params.numDataTypes,1);
    for d = 1:params.numDataTypes


    
        %training matrix and labels
        TrainingMatrix = SVMClassify.(params.data_types{d}).Predictors;
        G =  SVMClassify.(params.data_types{d}).Labels_ss{1};
    
        % train svm model
        clear M_TrialEigAcc AUC FalsePositiveRate TruePositiveRate prob_ts
        
        %train on random sub sample of data
        N = round(0.5 * size(TrainingMatrix,1) * 0.7);

        for iter = 1:params.svm.nIter

            AD_I = find(G ==1);
            AD_I = randsample(AD_I,N,'false');
            HC_I = find(G ==2);
            HC_I = randsample(HC_I,N,'false');
            
            DataTs = TrainingMatrix([AD_I,HC_I],:) ; 
            ConditionLabels = [ones(numel(AD_I),1);2 * ones(numel(HC_I),1)];
            
            NumConditions = max(ConditionLabels);
            for k=1:NumConditions
                % train svm
                model_ts = ...
                    fitcsvm(DataTs,ConditionLabels==k,'Standardize',true,...
                    'KernelFunction','linear','CrossVal','on');                
            
                % fit posterior distribution 
                [score_model_ts] = fitSVMPosterior(model_ts);
            
                % get posterior test prob
                [~, post_prob_ts] = kfoldPredict(score_model_ts);
            
                prob_ts(:,k) = post_prob_ts(:,2);  
            
            end
            
            % predictions
            [~,pred_ts] = max(prob_ts,[],2);

            % accuracy per class
            TrialEigAcc = [];
            for k = 1:NumConditions
                A = pred_ts(ConditionLabels ==k) == k;
            
                TrialEigAcc(k) = sum(A) ./ numel(A);
            end
            
            M_TrialEigAcc(iter,:) = TrialEigAcc;
            
            rocObj = rocmetrics(ConditionLabels,prob_ts,["1","2"]); %{'AD','HC'}
            
            TruePositiveRate(iter,1:2*N+1) = rocObj.Metrics.TruePositiveRate(1:2*N+1);
            FalsePositiveRate(iter,1:2*N+1) = rocObj.Metrics.FalsePositiveRate(1:2*N+1);
            AUC(iter,:) =  rocObj.AUC;

        end
            
        % rocObj.Metrics.TruePositiveRate = squeeze(mean(TruePositiveRate,1));
        % rocObj.Metrics.FalsePositiveRate = squeeze(mean(FalsePositiveRate,1));
        
        SVMResults{d}.Accuracy = M_TrialEigAcc;
        SVMResults{d}.AUC = AUC;
        SVMResults{d}.rocObj = rocObj;
        SVMResults{d}.TruePositiveRate = TruePositiveRate;
        SVMResults{d}.FalsePositiveRate =FalsePositiveRate;
        SVMResults{d}.groups= G;
    
        if params.svm.verbose
            disp(strcat(':: Info :: ', ...
                params.data_types{d},' mean accuracy :', ...
                num2str(mean(M_TrialEigAcc(:)))));
            disp(strcat(':: Info :: ', ...
                params.data_types{d},' std :', ...
                num2str(std(M_TrialEigAcc(:)))));
            disp(strcat(':: Info :: ', ...
                params.data_types{d},' AUC :', ...
                num2str(mean(AUC(:)))));
        end

    end
    
    save('SVMResults','SVMResults');
end

