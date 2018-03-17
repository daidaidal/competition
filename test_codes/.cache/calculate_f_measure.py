tp = 768
fp = 328
fn = 310
tn = 9671

P = tp/(tp+fp)
R = tp/(tp+fn)
F = 2*P*R/(P+R)

print("P:"+str(P))
print("R:"+str(R))
print("F:"+str(F))