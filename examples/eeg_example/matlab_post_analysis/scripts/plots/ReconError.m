function reconError = ReconError(params)

    clear reconError

    for s = 1:params.numSub

        params = config_loop(params,s);
    
        %load subject level flow fields
        clear FlowFields
        load(strcat('s',num2str(s),'_FlowFields'));


        for d = 1:params.numDataTypes

            % divergence
            divergence_temp{d} = ...
                normalize(FlowFields{d}.Div,1);
            
            %curl
            curl_temp{d}  = ...
                normalize(FlowFields{d}.Cav,1);

        end

        
        %compute error (curl)
         reconError.curl.spline(s) = ...
             squeeze(mean(abs(curl_temp{1} - curl_temp{2}),[1,2,3],'omitnan'));

         reconError.curl.rvgp(s) = ...
             squeeze(mean(abs(curl_temp{1} - curl_temp{3}),[1,2,3],'omitnan'));

         reconError.curl.linear(s) = ...
             squeeze(mean(abs(curl_temp{1} - curl_temp{4}),[1,2,3],'omitnan'));


        %compute error (divergence)
         reconError.divergence.spline(s) = ...
             squeeze(mean(abs(divergence_temp{1} - divergence_temp{2}),[1,2,3],'omitnan'));

         reconError.divergence.rvgp(s) = ...
             squeeze(mean(abs(divergence_temp{1} - divergence_temp{3}),[1,2,3],'omitnan'));

         reconError.divergence.linear(s) = ...
             squeeze(mean(abs(divergence_temp{1} - divergence_temp{4}),[1,2,3],'omitnan'));

    end

      

end
