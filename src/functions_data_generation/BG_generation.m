function BG = BG_generation(size, sigma2, ratio)
% Returns Bernulli-Gaussian outliers 
% Author: Lei Cheng

    BG = zeros(size);
    for k=1:prod(size)
        if (rand < ratio)
            BG(k)=(randn+1j*randn)*sqrt(0.5*sigma2);
        end
    end
end