function [blockSize, numThreads] = largestDivisor(numPoints, maxThreads)

quotients = numPoints ./ (1:maxThreads);
blockSize = find(abs(quotients - round(quotients)) < 1e-12, 1, 'last');
numThreads = round(quotients(blockSize));