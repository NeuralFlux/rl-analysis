import sys
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


def parse_line(line):
    args = line.split()
    epoch = int(args[0][1:-2])
    avg = float(args[3][:-1])
    return (epoch + 1), avg

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as log:
        lines = log.readlines()
        # print(lines)
    
    # print(lines[-23])
    # print(parse_line(lines[-23]))
    df = pd.DataFrame(data=map(parse_line, lines), columns=["epoch", "roll_avg"])
    df.drop_duplicates(subset="epoch", keep="last", inplace=True)
    df.reset_index(inplace=True, drop=True)
    # print(df.loc[df['epoch'] == 6540])

    # plot
    sns.lineplot(data=df.iloc[:638, :], x="epoch", y="roll_avg", label="lr=1e-3")
    sns.lineplot(data=df.iloc[638:, :], x="epoch", y="roll_avg", label="lr=1e-4")
    plt.legend()
    # plt.show()
    plt.savefig('checkpoints_vpg/learning_graph.png')
