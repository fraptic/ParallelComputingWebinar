function Xreal = gpufun2(Xreal, Ximag, creal, cimag)
%#codegen
coder.gpu.kernel;
for idx = 1:numel(Xreal)   
    ReX = Xreal(idx);
    ImX = Ximag(idx);
    for k = 1:100
        tmp = 2 * ReX * ImX + cimag;
        ReX = ReX * ReX - ImX * ImX + creal;
        ImX = tmp;
    end
    % Transformation
    Xreal(idx) = exp(-sqrt(ReX * ReX + ImX * ImX));
end



