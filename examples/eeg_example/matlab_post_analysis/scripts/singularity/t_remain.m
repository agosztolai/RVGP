
            
function t_remain(timeInSeconds,LoopsRemaining)

    %inputs
    % K : estimate of time duration per loop
    % N : number of loops remaining
    t = LoopsRemaining * timeInSeconds;

    % Convert to a date vector and then to a formatted string
    % Display the result
    % Convert to hours, minutes, and seconds
    hours = floor(t / 3600);
    minutes = floor(rem(t, 3600) / 60);
    seconds = round(rem(t, 60));


    
    timeStr = [num2str(hours),'h::', num2str(minutes),'m::' ,num2str(seconds),'s'];

    disp(strcat('Projected time remaining ... ',timeStr));


end
