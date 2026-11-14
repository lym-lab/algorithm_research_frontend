% Independent validation of published Kabal components at supplied energies.
% Run after scripts/export_reference.py; preserve vendor source unchanged.
project = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(project,'vendor','PEAQ_matlab','PQevalAudio')));
input = load(fullfile(project,'outputs','reference_input.mat'));
[Nc, fc] = PQCB('Basic');
reference_ear = PQWOME(input.frequencies);
[reference_a, reference_b] = PQtConst(.030,.008,input.centers,44100/512);
reference_internal = PQIntNoise(input.centers);
reference_spread = PQspreadCB(input.energy,'Basic');
save(fullfile(project,'outputs','reference_matlab.mat'),'reference_ear','reference_a','reference_internal','reference_spread','fc');
disp('Kabal reference values exported; Python comparison follows.');
