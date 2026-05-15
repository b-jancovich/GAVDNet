function test_egpu_throughput
% TEST_EGPU_THROUGHPUT  Measures host<->device bandwidth and link stability
% over the Thunderbolt connection to an external GPU. Run repeatedly; compare
% against expected TB3 throughput (~2.5 GB/s) and watch for variance.
%
% Ben Jancovich, 2026
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%

    g = gpuDevice;
    fprintf('Device: %s\n', g.Name);
    fprintf('Total memory: %.2f GB\n\n', g.TotalMemory/1e9);

    % 800 MB transfer per iteration (100M single-precision values)
    N = 1e8;
    nReps = 20;
    bytes = 4*N;

    h2d = zeros(nReps, 1);
    d2h = zeros(nReps, 1);

    data = rand(N, 1, 'single');

    for k = 1:nReps
        % Host -> Device
        wait(g);
        t0 = tic;
        d = gpuArray(data);
        wait(g);
        h2d(k) = bytes / toc(t0) / 1e9;

        % Device -> Host
        wait(g);
        t0 = tic;
        out = gather(d); %#ok<NASGU>
        wait(g);
        d2h(k) = bytes / toc(t0) / 1e9;

        fprintf('Iter %2d:  H->D %.2f GB/s   D->H %.2f GB/s\n', ...
            k, h2d(k), d2h(k));
    end

    fprintf('\nSummary:\n');
    fprintf('  H->D mean %.2f GB/s, std %.2f, min %.2f, max %.2f\n', ...
        mean(h2d), std(h2d), min(h2d), max(h2d));
    fprintf('  D->H mean %.2f GB/s, std %.2f, min %.2f, max %.2f\n', ...
        mean(d2h), std(d2h), min(d2h), max(d2h));

end