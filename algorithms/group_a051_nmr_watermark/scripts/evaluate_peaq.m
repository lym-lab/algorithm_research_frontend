function evaluate_peaq(pairs_csv, output_csv, vendor_directory)
% Run from any directory; Kabal uses relative addpath internally.
previous = pwd;
cleanup = onCleanup(@() cd(previous));
cd(vendor_directory);
addpath(genpath(vendor_directory));
pairs = readtable(pairs_csv, 'TextType','string', 'Delimiter',',', ...
    'ReadVariableNames',true, 'VariableNamingRule','preserve', 'Encoding','UTF-8');
odg = nan(height(pairs),1);
status = repmat("pending",height(pairs),1);
completed = 0;
checkpoint = [output_csv '.checkpoint.mat'];
if isfile(checkpoint)
    saved = load(checkpoint);
    if ~isequal(saved.pairs,pairs)
        error('PEAQ checkpoint input pairs differ');
    end
    odg = saved.odg;
    status = saved.status;
    completed = saved.completed;
end
for i = completed+1:height(pairs)
    % Capture verbose reference output without changing reference code.
    diagnostic = evalc('odg(i) = PQevalAudio(char(pairs.reference(i)), char(pairs.marked(i)));');
    if ~isfinite(odg(i))
        % Preserve undefined reference results; never replace them with a score.
        status(i) = "undefined_reference_output";
        file = sprintf('%s.undefined_%d.txt',output_csv,pairs.index(i));
        stream = fopen(file,'w');
        fprintf(stream,'%s',diagnostic);
        fclose(stream);
    else
        status(i) = "ok";
    end
    completed = i;
    if mod(i,25) == 0 || i == height(pairs)
        save([checkpoint '.tmp'],'pairs','odg','status','completed','-mat');
        movefile([checkpoint '.tmp'],checkpoint,'f');
        fprintf('PEAQ %d/%d; undefined %d\n',i,height(pairs),sum(status == "undefined_reference_output"));
    end
end
index = pairs.index;
writetable(table(index,odg,status),output_csv);
end
