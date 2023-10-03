% RGB_out = colormap_CD(hue,brt,gry,num)
% 
% By: Duo Chan, Harvard University
% duochan@g.harvard.edu
% Last Update: 2018-08-05
% 
% -------------------------------------------------------------------------
% Please read examples in README.md for quick start
% Input variables are:
% -------------------------------------------------------------------------
% 
% hue: hue of colors, valid values are from 0 to 1, which represent:
% 
% red --- orange --- yellow --- green --- cyan --- blue --- purple --- red
%  0       1/12       1/6        1/3       1/2      2/3      5/6        1     
% 
% The dimension of hue should be a n by m, where:
% n is number of groups of colors, which has no limitation:
%      when n = 1, it has one group of hues in the colormap
%      when n = 2, it has two groups of hues in the colormap
%      when n > 3, it has multiple groups of hues, which stack the colormap
%           of each group of hues by n times
% m represnt number of colors in each group, which can take values in {1,2}:
%      when m = 1, only one color is allowed in each group of hue
%      when m = 2, num number of colors in each group is used, and each hue
%           is determined by linear interpolation
%  
% -------------------------------------------------------------------------
% 
% brt: brightness of the colors, valid values are from 0 to 1, which is:
% 
% black --- color --- white
%   0        0.5        1
% 
% The dimention of hue should be 1x1 or 1x2,
%      when size(brt) = [1, 1], only one brightness is allowed in each
%           group of hue
%      when size(brt) = [1, 2], num number of brightness in each group
%           is used, and each brightness is determined by linear interpolation
% 
% -------------------------------------------------------------------------
% 
% gry: whether centain hues are in gray scale, valid values are {0, 1}
%      the size of gry should be 1xn, where n is number of groups of hue
% 
% -------------------------------------------------------------------------
% 
% num: number of colors in each group of hue

function RGB_out = colormap_CD(hue,brt,gry,num,do_disp)

    % *********************************************************************
    % If input size is one in the second dimension, repeat the matrix
    % *********************************************************************
    if size(hue,2)==1,         hue = [hue hue];    end
    if size(brt,2)==1,         brt = [brt brt];    end
    if size(brt,1)==1,         brt = repmat(brt,size(hue,1),1);    end
    
    % *********************************************************************
    % Round the data to have changes in hue always no greater than 0.5
    % *********************************************************************
    for ct = 1:size(hue,1)
        if (hue(ct,1) - hue(ct,2)) > 0.5,
            hue(ct,1) = hue(ct,1) - 1;
        end
        if (hue(ct,2) - hue(ct,1)) > 0.5,
            hue(ct,2) = hue(ct,2) - 1;
        end
    end
    
    % *********************************************************************
    % Default value for gry is zero
    % *********************************************************************
    if ~exist('gry','var'),    gry = zeros(size(hue,1),1);    end
    
    gry = gry(:);
    
    if numel(num) == 1,        num = repmat(num,size(hue,1),1);    end
    
    % *********************************************************************
    % Generate the RGB value for each group of hues
    % *********************************************************************
    for ct = 1:size(hue,1)
        
        if num(ct) == 1,
            col_intp = hue(ct,1);
            brt_intp = hue(ct,1);
        else
            col_intp = interp1([1 num(ct)],hue(ct,:),[1:num(ct)]);
            brt_intp = interp1([1 num(ct)],brt(ct,:),[1:num(ct)]);
        end
        
        col_intp(col_intp < 0) = col_intp(col_intp < 0) + 1;
        col_intp(col_intp > 1) = col_intp(col_intp > 1) - 1;
        
        mu = interp1([0 1/12 1/6 1/3 1/2 2/3 5/6 1],[.75 .85 .85 .7 .55 .8 .65 .75],col_intp,'spline');
        a  = 4 * (1-mu);
        gry_intp = a .* (brt_intp - 0.5).^2 + mu;
        if gry(ct) == 1, gry_intp(:) = 0; end

        RGB{ct} = colormap_CD_RGB([col_intp(:) gry_intp(:) brt_intp(:)]);
        
    end
    
    % *********************************************************************
    % Assemble and set the colormap
    % *********************************************************************
    if size(hue,1) == 1,
        RGB_out = RGB{1};
    elseif size(hue,1) == 2,
        RGB_out = [flipud(RGB{1}); RGB{2}];    
    else
        RGB_out = [];
        for i = 1:numel(RGB)
            RGB_out = [RGB_out; RGB{i}];
        end
    end
    

    if ~exist('do_disp','var'), do_disp = 1; end
    
    if do_disp ~= 0,
        colormap(gca,RGB_out);
    end

end

% *************************************************************************
% Subroutine that generate RGB values from HSB color representation
% *************************************************************************
function [output] = colormap_CD_RGB(Input)
    
    zzz = size(Input);
    if zzz(2) ~= 3,
        input = reshape(Input,zzz(1)*zzz(2),3);
    else
        input = Input;
    end
    clear('Input')

    Hue_temp = input(:,1);
    Str_temp = input(:,2);
    Brt_temp = input(:,3);

    Hue_temp = rem(Hue_temp + 1000,1);
    Hue_RGB = nan(size(input));

    Brt_RGB = nan(size(input));

    logic = Hue_temp>=0 & Hue_temp<1/6;
    Hue_RGB(logic,:)=repmat([1 0 0],nnz(logic),1) + repmat((Hue_temp(logic,:) - 0/6),1,3) .* 6 .* repmat([0 1 0],nnz(logic),1);

    logic = Hue_temp>=1/6 & Hue_temp<2/6;
    Hue_RGB(logic,:)=repmat([1 1 0],nnz(logic),1) - repmat((Hue_temp(logic,:) - 1/6),1,3) .* 6 .* repmat([1 0 0],nnz(logic),1);

    logic = Hue_temp>=2/6 & Hue_temp<3/6;
    Hue_RGB(logic,:)=repmat([0 1 0],nnz(logic),1) + repmat((Hue_temp(logic,:) - 2/6),1,3) .* 6 .* repmat([0 0 1],nnz(logic),1);

    logic = Hue_temp>=3/6 & Hue_temp<4/6;
    Hue_RGB(logic,:)=repmat([0 1 1],nnz(logic),1) - repmat((Hue_temp(logic,:) - 3/6),1,3) .* 6 .* repmat([0 1 0],nnz(logic),1);

    logic = Hue_temp>=4/6 & Hue_temp<5/6;
    Hue_RGB(logic,:)=repmat([0 0 1],nnz(logic),1) + repmat((Hue_temp(logic,:) - 4/6),1,3) .* 6 .* repmat([1 0 0],nnz(logic),1);

    logic = Hue_temp>=5/6 & Hue_temp<=6/6;
    Hue_RGB(logic,:)=repmat([1 0 1],nnz(logic),1) - repmat((Hue_temp(logic,:) - 5/6),1,3) .* 6 .* repmat([0 0 1],nnz(logic),1);

    Str_RGB = (Hue_RGB - .5) .* repmat(Str_temp,1,3) + .5;
         
    logic = Brt_temp > 0.5;
    Brt_RGB(logic,:) = Str_RGB(logic,:) + (1 - Str_RGB(logic,:)) .* repmat(((Brt_temp(logic,:)-0.5)./0.5),1,3);

    logic = Brt_temp <= 0.5;
    Brt_RGB(logic,:) = Str_RGB(logic,:) - (Str_RGB(logic,:)) .* repmat(((0.5 - Brt_temp(logic,:))./0.5),1,3);

    output = Brt_RGB;

    if zzz(2) ~= 3,
        output = reshape(output,zzz(1),zzz(2),3);
    end
    
    output(output>1) = 1;
    output(output<0) = 0;
end