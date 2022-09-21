from neural_network.neural_networks import *
from util import *


def neural_networks_ufc_test():
    X, y, X_test, y_test, X_future = construct_data()

    rows, columns = X.shape
    X = np.concatenate([np.ones((rows, 1)),
                        X], axis=1)
    X_future = np.concatenate([np.ones((X_future.shape[0], 1)),
                               X_future], axis=1)
    X_test = np.concatenate([np.ones((X_test.shape[0], 1)),
                             X_test], axis=1)
    y = y.reshape(y.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    print("----------NNFS---------------")
    model = Model()
    model.add(Layer_Dense(X.shape[1], 512, weight_regularizer_l2=1e-4,
                          bias_regularizer_l2=1e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(512, 512, weight_regularizer_l2=1e-4,
                          bias_regularizer_l2=1e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(512, 512, weight_regularizer_l2=1e-4,
                          bias_regularizer_l2=1e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(512, 1))
    model.add(Activation_Sigmoid())

    model.set(loss=Loss_BinaryCrossentropy(),
              optimizer=Optimizer_Adam(decay=1e-9), accuracy=Accuracy_Categorical(binary=True))
    model.finalize()
    model.train(X, y, epochs=60, print_every=50)

    model.evaluate(X_test, y_test)

    print("After some testing, the following parameters worked best for the UFC dataset:")
    print("Neural Network: input(21,512) l2 regulazier = 1e-4 -> dense layer (512,512) l2 regulazier = 1e-4 -> dense layer (512,512) l2 regulazier = 1e-4 -> dense layer(512,1)")
    print("Optimizer Adam with decay 1e-9 - epochs 60")


def main():
    neural_networks_ufc_test()


if __name__ == "__main__":
    main()
