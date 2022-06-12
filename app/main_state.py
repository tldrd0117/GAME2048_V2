from typing import List
from repo.tree import TreeDbRepository
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datasource.mongo import MongoDataSource

tree = TreeDbRepository()

data = tree.getGameInfo("TrainMultiAiServiceAverageLRChange")
# data = tree.getGameInfo("TrainMultiAiServiceLRChange")
print(f"Epoch Length:{len(data)}")

db = MongoDataSource()


def plotLosses():
    mpl.style.use("default")
    cursor = db.getLosses()
    losses = []
    for d in list(cursor):
        losses.append(float(d["loss"]))
    print(losses)
    fig, ax = plt.subplots()
    ax.plot(list(range(len(losses))), losses, label="loss")
    plt.show()


def toggle_plot(line):
    line.set_visible(not line.get_visible())
    plt.draw()

def average(lst):
    if len(lst) <= 0:
        return 0
    return sum(lst) / len(lst)

def averagePlot(ax, dates, data, dataName, q):
    average5Data = []
    average20Data = []
    average100Data = []
    average200Data = []
    for index in range(len(data)):
        slice1 = data[(index-5):index]
        slice2 = data[(index-20):index]
        slice3 = data[(index-100):index]
        slice4 = data[(index-200):index]
        average5Data.append(average(slice1))
        average20Data.append(average(slice2))
        average100Data.append(average(slice3))
        average200Data.append(average(slice4))
    datas = [data, average5Data, average20Data, average100Data, average200Data]
    btnNames = [dataName, dataName+"_5", dataName+"_20", dataName+"_100", dataName+"_200"]
    lines = []
    axes = []
    btns = []
    for i in range(5):
        line, = ax.plot(dates, datas[i], label=btnNames[i])
        line.set_visible(not line.get_visible())
        lines.append(line)
    for i in range(5):
        axe = plt.axes([i/10+0.01, 0.92 - (0.075 * (q)), 0.1, 0.075])
        axes.append(axe)
    for i in range(5):
        btn = Button(axes[i], btnNames[i])
        btns.append(btn)
    return btns, lines

def applyEvent(btns, lines):
    btns[0].on_clicked(lambda e : toggle_plot(lines[0]))
    btns[1].on_clicked(lambda e : toggle_plot(lines[1]))
    btns[2].on_clicked(lambda e : toggle_plot(lines[2]))
    btns[3].on_clicked(lambda e : toggle_plot(lines[3]))
    btns[4].on_clicked(lambda e : toggle_plot(lines[4]))


def plot():
    # fig, ax = plt.subplots()  # Create a figure containing a single axes.
    dates = []
    maxQ = []
    scores = []
    turns = []
    mpl.style.use("default")
    for one in data:
        dates.append(one["createdAt"])
        scores.append(one["score"])
        turns.append(one["turn"])
        if "maxQ" in one:
            maxQ.append(one["maxQ"])
        else:
            maxQ.append(0)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    y = range(len(data))
    btns, lines = averagePlot(ax, y, maxQ, "maxQ", 0)
    btns2, lines2 = averagePlot(ax2, y, scores, "scores", 1)
    btns3, lines3 = averagePlot(ax2, y, turns, "turns", 2)
    applyEvent(btns, lines)
    applyEvent(btns2, lines2)
    applyEvent(btns3, lines3)
    

    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1)
    plt.show()

def getAverageByDay():
    def getAverage(li: List):
        length = len(li)
        turn = 0
        score = 0
        maxQ = 0
        maxQLength = 0
        for one in li:
            turn = turn + one["turn"]
            score = score + one["score"]
            if "maxQ" in one:
                maxQ = maxQ + one["maxQ"]
                maxQLength = maxQLength + 1
        return {
            "turn": turn / length,
            "score": score/ length,
            "length": length,
            "maxQ": maxQ / maxQLength if maxQLength > 0 else 0,
            "maxQLength": maxQLength
        }

    infoDict = {}
    for one in data:
        now = one["createdAt"]
        key = f"{now.year}, {now.month}, {now.day}, {now.hour}"
        if key not in infoDict:
            infoDict[key] = []
        infoDict[key].append(one)

    for key in infoDict.keys():
        print(f"key: {key} data: {str(getAverage(infoDict[key]))}")
plot()
# plotLosses()
# poetry run python app/main_state.py