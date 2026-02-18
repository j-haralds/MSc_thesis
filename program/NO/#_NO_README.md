# Versions and experimental log for Neural Operators (NOs)

### Version 0
Included a basic example w/ an FNO predicting the derivative of a sinusoid

### Version 1
A basic FNO taking $I(t)$ as input and outputting $V(t)$

### Version 2

Trying a basic FNO on a concentration profile. Using $I(t)$ as input and predicting the full concentration profile. 

### Version 3
I'll use $I(t), r, t$ as inputs and then try, by using AD, to write a PI-FNO with the diffusion equation as the loss. Taking the negative electrode first. I verified that $D_p\equiv\mathrm{const.}$ I wonder if one potentially could employ unsupervised learning in this case. 
