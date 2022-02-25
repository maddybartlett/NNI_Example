import nni
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
        test_loss = float(test_loss.detach().numpy())
        nni.report_intermediate_result(test_loss)
    nni.report_final_result(test_loss)

if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)