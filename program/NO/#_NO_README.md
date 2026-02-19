# Versions and experimental log for Neural Operators (NOs)

### Version 0
Included a basic example w/ an FNO predicting the derivative of a sinusoid

### Version 1
A basic FNO taking $I(t)$ as input and outputting $V(t)$

### Version 2

Trying a basic FNO on a concentration profile. Using $I(t)$ as input and predicting the full concentration profile. 

### Potential Version 2.1 (NOT STARTED)
I'll use $I(t), r, t$ as inputs and then try, by using AD, to write a PI-FNO with the diffusion equation as the loss. Taking the negative electrode first. I verified that $D_p\equiv\mathrm{const.}$ I wonder if one potentially could employ **unsupervised** learning in this case. 

### Version 3 
Encode the SPM equations in the FNO. NO takes $I(t)$ and returns $c(r,t)$, $U_n$, $U_p$ as a hidden state. Then, employing $$V=U_{eq}-\eta_r$$
$$U_{eq} = U_p(c_p|_{r=R_p})-U_n(c_n|_{r=R_n})$$ $$\eta_r = \frac{2RT}{F}\Bigg[\mathrm{arcsinh}\left(\frac{j_n(t)}{j_{n,0}(t)}\right)-\mathrm{arcsinh}\left(\frac{j_p(t)}{j_{p,0}(t)}\right)\Bigg],$$
$$j_{k,0}(t) = FK_k\sqrt{\frac{c_k(r,t)}{c_k^\mathrm{max}}\left(1-\frac{c_k(r,t)}{c_k^\mathrm{max}}\right)}\Bigg|_{r = R_k}\text{for }k\in\{n,p\}$$
