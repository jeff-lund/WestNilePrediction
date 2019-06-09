import matplotlib.pyplot as plt
import os

def plot_loss(history, title, out_dir, offset):
    '''
    creates a plot of loss vs epochs
    history is the history field from the history object
    returned from fitting a model
    '''
    training_loss = [x * 100 for x in history['loss'][::offset]]
    testing_loss = [x * 100 for x in history['val_loss'][::offset]]
    x_axis = [x * offset for x in range(len(training_loss))]
    plt.plot(x_axis, training_loss, label="Training")
    plt.plot(x_axis, testing_loss, label="Test")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    sv_title = title.replace(' ', '_')
    sv_title = os.path.join(out_dir, sv_title)
    plt.savefig(sv_title + '_loss.png', format='png', bbox_inches='tight')
    plt.close()
