import torch
from utils import DataLoad, train, draw_graph, final_train, test
from model import SimpleCNN_M3, SimpleCNN_M5, SimpleCNN_M7


def main(batch_size, epochs, learning_rate, file_dir=''):
    Dataset = DataLoad(file_dir=file_dir, batch_size=batch_size)

    train_loader = Dataset.train_data_load()
    valid_loader = Dataset.valid_data_load()
    test_loader = Dataset.test_data_load()

    model = SimpleCNN_M7()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_idx, train_acc, train_loss, valid_acc, valid_loss = train(optimizer=optimizer, epochs=epochs,
                                                                   model=model, train_loader=train_loader,
                                                                   valid_loader=valid_loader,
                                                                   criterion=criterion)

    draw_graph(best_idx, train_acc, train_loss, valid_acc, valid_loss)

    model = SimpleCNN_M7()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model = final_train(optimizer=optimizer, epochs=best_idx,
                        model=model, train_loader=train_loader,
                        valid_loader=valid_loader,
                        criterion=criterion)

    test(model=model, test_loader=test_loader)


if __name__ == '__main__':
    BATCH_SIZE = 120
    EPOCHS = 150
    LEARNING_RATE = 0.001
    FILE_DIR = './data/'
    main(BATCH_SIZE, EPOCHS, LEARNING_RATE, FILE_DIR)
