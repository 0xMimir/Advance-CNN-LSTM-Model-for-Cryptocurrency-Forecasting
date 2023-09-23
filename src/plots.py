from matplotlib.pyplot import plot, title, ylabel, xlabel, legend, savefig

def plot_history(history: dict, value: str):
    plot(history[value])
    plot(history[f'val_{value}'])

    title('Training history')
    ylabel(value)
    xlabel('epoch')

    legend(['train', 'validation'], loc='upper left')
    savefig('plot.png')