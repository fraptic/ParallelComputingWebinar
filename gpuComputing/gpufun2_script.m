% GPUFUN2_SCRIPT   Generate MEX-function gpufun2_mex from gpufun2.
% 
% Script generated from project 'gpufun2.prj' on 09-Jun-2020.
% 
% See also CODER, CODER.CONFIG, CODER.TYPEOF, CODEGEN.

%% Create configuration object of class 'coder.MexCodeConfig'.
cfg = coder.gpuConfig('mex');
cfg.InitFltsAndDblsToZero = false;
cfg.GenerateReport = true;
cfg.SaturateOnIntegerOverflow = false;
cfg.IntegrityChecks = true;
cfg.ResponsivenessChecks = true;
cfg.GpuConfig.Enabled = false;
cfg.GpuConfig.SafeBuild = true;

%% Define argument types for entry-point 'gpufun2'.
ARGS = cell(1,1);
ARGS{1} = cell(4,1);
ARGS{1}{1} = coder.typeof(0,[2000 4000],'Gpu',true);
ARGS{1}{2} = coder.typeof(0,[2000 4000],'Gpu',true);
ARGS{1}{3} = coder.typeof(0);
ARGS{1}{4} = coder.typeof(0);

%% Invoke MATLAB Coder.
codegen -config cfg gpufun2 -args ARGS{1}

