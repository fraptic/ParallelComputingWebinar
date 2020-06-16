function Z = juliaCalc(Z,c)

for k = 1:100
    Z = Z.^2 + c;
end
Z = exp(-abs(Z));
