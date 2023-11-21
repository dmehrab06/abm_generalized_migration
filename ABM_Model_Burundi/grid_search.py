import sys

total = 10
range_left = int(sys.argv[1])
range_right = range_left+total

val_left = float(sys.argv[2])
val_right = float(sys.argv[3])

comment = str(sys.argv[4])

step_size = (val_right-val_left)/(total-1)

for i in range(range_left,range_right):
    if i==range_left:
        print('python pipeline_gen.py',i,comment,round(val_left,2),'>',comment+'.sh')
    else:
        print('python pipeline_gen.py',i,comment,round(val_left,2),'>>',comment+'.sh')
    val_left = val_left + step_size