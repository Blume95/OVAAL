ovaal_cs = {1: 0.5277056226198169, 4: 0.6283450888472899, 6: 0.6429334209109357,
            10: 0.6555551942098119}
ovaal_cs = {1: 0.20047432079094096, 4: 0.2723965797996301, 6: 0.30245862177890115,
            10: 0.3084578463019059}

pixelpick_cs = {1: 0.5569032053320945, 4: 0.5681922175129386, 6: 0.5702162096361031, 10: 0.58855763526555908}
pixelpick_cs = {1: 0.19006016433900877, 4: 0.1925258982001986, 6: 0.19308999329409163, 10: 0.19429337632896868}
import matplotlib.pyplot as plt

x_ = [x * (11 / (260 * 480)) * 100 for x in pixelpick_cs.keys()]
y_ovaal = []
y_pixelpick = []
for k, v in ovaal_cs.items():
    y_ovaal.append(v)
for k, v in pixelpick_cs.items():
    y_pixelpick.append(v)

plt.plot(x_, y_ovaal, label="OVAAL")
plt.plot(x_, y_pixelpick, label="Pixelpick")
# plt.plot(x_[-1], 0.6629958817563388, "*", label='OVAAL+')
plt.xlabel("% Pixels of One Image")
plt.ylabel("mIoU")
plt.legend()
plt.title("Kitti")
plt.show()
