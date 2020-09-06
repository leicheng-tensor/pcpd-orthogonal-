function W = noise_generation(SNR, X)
% return Gaussian noise with SNR defined as SNR = 10*log_10^{\frac{|X|^2}{|W|^2}}
% Author: Lei Cheng
c=10^(SNR/10);
W= sqrt(0.5)*(randn(size(X))+1j*randn(size(X)));
U_X1=double(tenmat(X,1));
U_W= double(tenmat(tensor(W),1));
eta=norm(U_X1,'fro')/(norm(U_W,'fro')*sqrt(c));
W=W*eta;
end