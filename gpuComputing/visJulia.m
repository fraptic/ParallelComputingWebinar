function visJulia(x,y,Z)
% VISJULIA is a helper function for visualizing a Julia set
figure('Color', 'w')
axes('Units', 'normalized', 'Position', [0 0 1 1])
cmap = parula(256);
cmap(1,:) = [1 1 1];
colormap(cmap)
imagesc(x,y,Z)
axis off