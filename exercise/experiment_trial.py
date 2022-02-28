from data_fetch import DividedPlane
from net import RobustNetwork

def main(args):
    # load data
    train = DividedPlane(n=200, noise=0.2, seed=165)
    test = DividedPlane(n=5000)
    # build model
    model = RobustNetwork(nodes=[2, args['nodes1'], args['nodes2'], 1])
    # train
    for epoch in range(10):
        model.learn(train.inputs(), train.targets(), epochs = 100, lr=args['lr'], weight_decay=args['weight_decay'])
        y = model(test.inputs())
        test_loss = model.loss_fcn(y, test.targets())
        test_loss = test_loss.item()
        print(test_loss)
    print(test_loss)

if __name__ == '__main__':
    params = {'nodes1': 20, 'nodes2': 30, 'lr': 0.001, 'weight_decay': 0.5}
    main(params)