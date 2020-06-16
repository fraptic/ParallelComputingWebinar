% RUNBIRTHDAY_CODEGEN_SCRIPT   Generate MEX-function runBirthday_mex from
%  runBirthday.
% 
% Script generated from project 'runBirthday.prj' on 16-Jun-2020.
% 
% See also CODER, CODER.CONFIG, CODER.TYPEOF, CODEGEN.

%% Create configuration object of class 'coder.MexCodeConfig'.
cfg = coder.config('mex');
cfg.MATLABSourceComments = true;
cfg.GenerateReport = true;
cfg.ReportPotentialDifferences = false;
cfg.SaturateOnIntegerOverflow = false;
cfg.IntegrityChecks = false;
cfg.ResponsivenessChecks = false;
cfg.ExtrinsicCalls = false;

%% Define argument types for entry-point 'runBirthday'.
ARGS = cell(1,1);
ARGS{1} = cell(2,1);
ARGS{1}{1} = coder.typeof(0);
ARGS{1}{2} = coder.typeof(0);

%% Invoke MATLAB Coder.
codegen -config cfg runBirthday -args ARGS{1}

