function checkGPU
% CHECKGPU checks if the system has a supported GPU

if gpuDeviceCount == 0
        error('juliademo:nogpu',[...
        'This system''s GPU is not supported by MATLAB. \n' ,...
        'For more information, please click ', ...
        '<a href="https://www.mathworks.com/products/availability.html#DM">here</a>. \n', ...
        '(scroll down to "Requirements for GPU Computing" if needed)']);
end