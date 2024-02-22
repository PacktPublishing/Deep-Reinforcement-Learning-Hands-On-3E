from lib import *


if __name__ == "__main__":
    net = DQNNet()
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    print("Main net:", net.ff.weight)
    print("Target net:", tgt_net.target_model.ff.weight)
    net.ff.weight.data += 1.0
    print("After update")
    print("Main net:", net.ff.weight)
    print("Target net:", tgt_net.target_model.ff.weight)
    tgt_net.sync()
    print("After sync")
    print("Main net:", net.ff.weight)
    print("Target net:", tgt_net.target_model.ff.weight)
