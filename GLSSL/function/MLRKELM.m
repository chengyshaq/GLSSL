function [Outputs,Pre_Labels] = MLRKELM(X,Y,Xt,Yt,parameter)
C = parameter.C;
Kpara = parameter.Kpara;

[OutputWeight,Omega_test,Y] = kelmtrain (X, Y, Xt, C, Kpara);
TY = kelmpredict (OutputWeight,Omega_test);

Outputs = TY';
Pre_Labels = sign(Outputs);
