fa_threshold = 1e-4  #虚警率阈值          False alarm rate threshold
alpha = 0.5          #检测精度Sp的加权系数 The weighting coefficient of the performance score Sp
Pbase = 2.225          #基线的参数量 M      The parameters for the baseline (Unit: M)
Fbase = 12.56         #基线的运算量 GFlops The FLOPs for the baseline


# 输入模型测试得到的评价指标 Input evaluation indicators obtained from model testing
Fa = 1.913179e-05 #虚警率 False alarm rate
Pd = 0.5555   #检测率 Detection rate
IoU = 0.36217  #平均交并比 average intersection over union
Params = 0.914057 #模型参数量 M Parameters of your model (Unit: M)
Flops = 5.1789169 #模型运算量 GFlops of your model


#评分 Score
# Sp = alpha*IoU + (1-alpha)*Pd
if Fa > fa_threshold:#
    Sp = 0
else:
    Sp = alpha * IoU + (1 - alpha) * Pd

# Se
Se = 1 - ((Params/Pbase + Flops/Fbase) * 0.5)

# Spe = (Se +Sp)/2
Spe = (Sp + Se) *0.5

print("IoU: {:.3f}, Fa: {}, Pd: {:.3f}, Sp: {}".format(IoU, Fa, Pd, Sp))
print("Params: {:.3f}M , FLOPs: {:.3f}GFlops, Se: {}".format(Params, Flops, Se))
print("Spe: ",format(Spe))
