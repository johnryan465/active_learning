import matplotlib.pyplot as plt
import pandas

spacing = 5
start = 20

x1 = []
y1 = []
x2 = []
y2 = []
for i in range(0, 30):
    csv = pandas.read_csv("experiments/vduq_bb_more_epochs_rerun/test-" + str(i) + ".csv")
    x1.append(start + (i*spacing))
    y1.append(csv["accuracy"].max())


for i in range(0, 10):
    csv = pandas.read_csv("experiments/random_more_epochs_rerun/test-" + str(i) + ".csv")
    x2.append(start + (i*spacing))
    y2.append(csv["accuracy"].max())

fig, ax = plt.subplots()
ax.plot(x1, y1, label="BB")
ax.plot(x2, y2, label="Random")
ax.set_title('Dataset size vs accuracy')

plt.savefig('comparision.png')
