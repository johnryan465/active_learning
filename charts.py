import json
import matplotlib.pyplot as plt
# read file

x = {}
with open('../experiment_state-2021-02-26_21-23-27.json', 'r') as myfile:
    data=myfile.read()
    obj = json.loads(data)
    # print(obj['checkpoints'])
    for i in obj['checkpoints']:
        c = i['config']
        m = c['method']
        del c['method']
        if str(c) not in x.keys():
            x[str(c)] = []
        x[str(c)].append((m,i['last_result'].get('accuracy')))

print(x)
for run in x:
    print(run, x[run])
    plt.plot(data=x[run])
plt.savefig('foo.png')
